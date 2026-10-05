"""Tests sans réseau : catalogue temporaire, ranker entraîné rapidement et faux LLM."""
from __future__ import annotations

from collections import deque
from datetime import date

import numpy as np
import pandas as pd
import pytest

from travel_agent import catalog, tools
from travel_agent.agent import TravelAgent
from travel_agent.ranking import (FEATURES, HeuristicBaseline, PairwiseLogReg, build_features, mrr,
                                  ndcg_at_k, simulate_relevance)
from travel_agent.schemas import Recommendation, TravelRequest


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    db = catalog.build(tmp_path_factory.mktemp("db") / "travel.db")
    return tools.connect(db)


REQ = TravelRequest(origin="PAR", destination="LIS", date_from=date(2027, 5, 21), date_to=date(2027, 5, 31),
                    preference="cheap", max_stops=1, nights=2, hotel_max_price=200)


class FakeLLM:
    def __init__(self, responses):
        self.q = deque(responses)

    def chat_json(self, system, user, schema):
        out = self.q.popleft()
        assert isinstance(out, schema)
        return out


def test_sql_filters_respect_constraints(con):
    df = tools.search_flights(con, REQ.model_copy(update={"max_price": 150}))
    assert len(df) > 0
    assert (df["price_eur"] <= 150).all() and (df["stops"] <= 1).all()
    assert df["depart_date"].between("2027-05-21", "2027-05-31").all()


def test_metrics():
    y = np.array([3, 2, 0, 1])
    assert ndcg_at_k(y, np.array([4, 3, 2, 1.5]), 4) < 1.0
    assert ndcg_at_k(y, np.array([4, 3, 1, 2]), 4) == pytest.approx(1.0)
    assert mrr(y, np.array([0, 1, 2, 3])) == pytest.approx(0.25)


def test_learned_ranker_beats_price_on_fast_users(con):
    rng = np.random.default_rng(0)
    frames = []
    for qid in range(60):
        req = REQ.model_copy(update={"preference": "fast", "max_stops": None})
        c = tools.search_flights(con, req).sample(30, random_state=qid)
        f = build_features(c, req)
        f["relevance"], f["qid"] = simulate_relevance(f, "fast", rng), qid
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    train, test = df[df.qid < 45], df[df.qid >= 45]
    model = PairwiseLogReg().fit(train[FEATURES], train["relevance"].to_numpy(), train["qid"].to_numpy())
    learned = np.mean([ndcg_at_k(q["relevance"].to_numpy(), model.score(q[FEATURES])) for _, q in test.groupby("qid")])
    price = np.mean([ndcg_at_k(q["relevance"].to_numpy(), -q["log_price_rel"].to_numpy()) for _, q in test.groupby("qid")])
    assert learned > price + 0.2


def test_agent_full_path_and_hallucination_guard(con):
    llm = FakeLLM([REQ, Recommendation(flight_ids=[999999999], message="Voici mes choix.")])
    agent = TravelAgent(llm, con=con, ranker=HeuristicBaseline())
    ans = agent.run("Paris Lisbonne fin mai pas cher, 2 nuits")
    assert ans.status == "ok" and len(ans.flights) == 3   # repli sur le top du ranker
    assert any("garde-fou" in t for t in ans.trace)
    assert len(ans.hotels) > 0 and all(h["price_night_eur"] <= 200 for h in ans.hotels)
    assert agent.cities["PAR"] == "Paris"


def test_agent_asks_for_clarification(con):
    llm = FakeLLM([TravelRequest(destination="LIS")])
    ans = TravelAgent(llm, con=con, ranker=HeuristicBaseline()).run("Un billet pour Lisbonne")
    assert ans.status == "clarification" and "départ" in ans.message


def test_agent_rejects_unknown_city_code(con):
    llm = FakeLLM([TravelRequest(origin="XXX", destination="LIS", date_from=date(2027, 5, 3))])
    ans = TravelAgent(llm, con=con, ranker=HeuristicBaseline()).run("Tombouctou Lisbonne le 3 mai")
    assert ans.status == "clarification"


def test_agent_no_result(con):
    llm = FakeLLM([REQ.model_copy(update={"max_price": 1})])
    ans = TravelAgent(llm, con=con, ranker=HeuristicBaseline()).run("Paris Lisbonne pour 1 euro")
    assert ans.status == "no_result"
