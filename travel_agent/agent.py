"""Agent de recherche de voyages orchestré avec LangGraph.

    parse ──(champs manquants)──> clarify ──> FIN
      │
      └──> search_flights (SQL) ──(0 résultat)──> no_result ──> FIN
                 │
                 └──> rank (modèle appris) ──> search_hotels ──> respond (LLM, ancré) ──> FIN

Le LLM intervient deux fois : pour transformer la demande en requête structurée, puis pour
rédiger la recommandation à partir des seules options classées. Les identifiants cités par le
LLM sont vérifiés : un identifiant inventé est rejeté (garde-fou contre les hallucinations).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import TypedDict

import joblib
import pandas as pd
from langgraph.graph import END, StateGraph

from .llm import LLM
from .ranking import build_features
from .schemas import AgentAnswer, Recommendation, TravelRequest
from .tools import city_codes, connect, search_flights, search_hotels

REFERENCE_DATE = date(2027, 3, 15)  # « aujourd'hui » pour interpréter les dates relatives
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ranker.joblib"
SHORTLIST = 5

PARSE_PROMPT = """Tu convertis une demande de voyage en requête structurée.
Date de référence (aujourd'hui) : {today}. Le catalogue couvre les départs du 2027-04-01 au 2027-05-31.
Villes disponibles (code: nom) : {cities}.
Règles :
- Utilise uniquement les codes de villes ci-dessus. Si une ville n'est pas mentionnée, mets null.
- Une date précise donne date_from = date_to. « Début mai » = 2027-05-01 à 2027-05-10,
  « mi-mai » = 2027-05-11 à 2027-05-20, « fin mai » = 2027-05-21 à 2027-05-31, un week-end = vendredi-samedi.
- preference : cheap (moins cher, petit budget), fast (rapide, direct, court), comfort (confort,
  bonne compagnie), sinon balanced. « vol direct » implique max_stops = 0.
- time_window : morning (matin), afternoon (après-midi), evening (soir), sinon any.
- nights : nombre de nuits d'hôtel seulement si un hôtel est demandé."""

RESPOND_PROMPT = """Tu es un assistant de voyage. Recommande des options UNIQUEMENT parmi celles fournies
(elles sont déjà classées par un modèle de ranking, la première est la mieux classée).
N'invente ni prix, ni horaire, ni identifiant. Explique en 3 à 6 phrases pourquoi ces options
correspondent à la demande (prix, durée, escales, horaire), et mentionne un compromis si pertinent."""


class AgentState(TypedDict, total=False):
    query: str
    request: TravelRequest
    flights: pd.DataFrame
    ranked: pd.DataFrame
    hotels: pd.DataFrame
    answer: AgentAnswer
    trace: list[str]


def _fmt_hour(h: float) -> str:
    return f"{int(h):02d}h{int(round((h % 1) * 60)):02d}"


class TravelAgent:
    def __init__(self, llm: LLM, con: sqlite3.Connection | None = None, ranker=None):
        self.llm = llm
        self.con = con or connect()
        self.ranker = ranker or joblib.load(MODEL_PATH)
        self.cities = city_codes(self.con)
        self.graph = self._build()

    # --- nœuds -------------------------------------------------------------
    def parse(self, s: AgentState) -> AgentState:
        cities = ", ".join(f"{k}: {v}" for k, v in self.cities.items())
        req = self.llm.chat_json(PARSE_PROMPT.format(today=REFERENCE_DATE, cities=cities), s["query"],
                                 TravelRequest)
        for field in ("origin", "destination"):  # validation : codes inconnus => champ manquant
            if getattr(req, field) and getattr(req, field) not in self.cities:
                setattr(req, field, None)
        return {"request": req, "trace": s.get("trace", []) + [f"parse: {req.model_dump_json(exclude_none=True)}"]}

    def clarify(self, s: AgentState) -> AgentState:
        labels = {"origin": "la ville de départ", "destination": "la destination", "date_from": "les dates"}
        missing = [labels[m] for m in s["request"].missing_fields()]
        msg = "Pour lancer la recherche, pouvez-vous préciser " + ", ".join(missing) + " ?"
        return {"answer": AgentAnswer(status="clarification", message=msg, request=s["request"],
                                      trace=s["trace"] + ["clarify"])}

    def search(self, s: AgentState) -> AgentState:
        flights = search_flights(self.con, s["request"])
        return {"flights": flights, "trace": s["trace"] + [f"search_flights: {len(flights)} candidats"]}

    def no_result(self, s: AgentState) -> AgentState:
        msg = ("Aucun vol ne correspond à ces critères. Essayez d'élargir les dates, "
               "d'augmenter le budget ou d'accepter une escale.")
        return {"answer": AgentAnswer(status="no_result", message=msg, request=s["request"],
                                      trace=s["trace"] + ["no_result"])}

    def rank(self, s: AgentState) -> AgentState:
        flights = s["flights"].copy()
        flights["score"] = self.ranker.score(build_features(flights, s["request"]))
        ranked = flights.sort_values("score", ascending=False).head(SHORTLIST)
        return {"ranked": ranked, "trace": s["trace"] + [f"rank: top={ranked['id'].tolist()}"]}

    def hotels(self, s: AgentState) -> AgentState:
        if not s["request"].nights:
            return {"hotels": pd.DataFrame()}
        h = search_hotels(self.con, s["request"])
        return {"hotels": h, "trace": s["trace"] + [f"search_hotels: {len(h)} hôtels"]}

    def respond(self, s: AgentState) -> AgentState:
        ranked, hotels = s["ranked"], s["hotels"]
        options = ranked.assign(depart=ranked["depart_hour"].map(_fmt_hour))[
            ["id", "depart_date", "depart", "duration_min", "stops", "airline", "price_eur"]]
        payload = {"demande": s["query"], "vols_classes": options.to_dict(orient="records"),
                   "hotels": hotels.to_dict(orient="records") if len(hotels) else []}
        rec = self.llm.chat_json(RESPOND_PROMPT, json.dumps(payload, ensure_ascii=False, default=str),
                                 Recommendation)
        trace = s["trace"]
        valid = [i for i in rec.flight_ids if i in set(ranked["id"])]
        if len(valid) < len(rec.flight_ids):
            trace = trace + [f"garde-fou: identifiants inventés ignorés {set(rec.flight_ids) - set(valid)}"]
        valid = valid or ranked["id"].tolist()[:3]
        flights = ranked.set_index("id").loc[valid].reset_index().to_dict(orient="records")
        answer = AgentAnswer(status="ok", message=rec.message, request=s["request"], flights=flights,
                             hotels=hotels.to_dict(orient="records") if len(hotels) else [],
                             trace=trace + ["respond"])
        return {"answer": answer}

    # --- graphe ------------------------------------------------------------
    def _build(self):
        g = StateGraph(AgentState)
        for name in ("parse", "clarify", "search", "no_result", "rank", "hotels", "respond"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("parse")
        g.add_conditional_edges("parse", lambda s: "clarify" if s["request"].missing_fields() else "search",
                                {"clarify": "clarify", "search": "search"})
        g.add_conditional_edges("search", lambda s: "no_result" if s["flights"].empty else "rank",
                                {"no_result": "no_result", "rank": "rank"})
        g.add_edge("rank", "hotels")
        g.add_edge("hotels", "respond")
        for end in ("clarify", "no_result", "respond"):
            g.add_edge(end, END)
        return g.compile()

    def run(self, query: str) -> AgentAnswer:
        return self.graph.invoke({"query": query, "trace": []})["answer"]
