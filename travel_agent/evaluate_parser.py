"""Évalue l'extraction de requêtes structurées par le LLM sur un jeu annoté à la main.

    python -m travel_agent.evaluate_parser

Métriques : exactitude par champ, taux de requêtes parfaitement extraites, et détection correcte
des champs manquants (qui déclenchent une question de clarification).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .agent import PARSE_PROMPT, REFERENCE_DATE
from .catalog import CITIES
from .llm import LLMClient
from .schemas import TravelRequest

ROOT = Path(__file__).resolve().parents[1]


def compare(pred: TravelRequest, gold: dict) -> dict[str, bool]:
    gold_req = TravelRequest(**gold)  # complète les valeurs par défaut
    return {f: getattr(pred, f) == getattr(gold_req, f) for f in TravelRequest.model_fields}


def main() -> None:
    load_dotenv()
    llm = LLMClient()
    items = [json.loads(line) for line in (ROOT / "data" / "parser_eval.jsonl").read_text().splitlines()]
    cities = ", ".join(f"{c[0]}: {c[1]}" for c in CITIES)
    system = PARSE_PROMPT.format(today=REFERENCE_DATE, cities=cities)
    rows, latencies = [], []
    for it in items:
        t0 = time.perf_counter()
        pred = llm.chat_json(system, it["query"], TravelRequest)
        latencies.append(time.perf_counter() - t0)
        ok = compare(pred, it["gold"])
        rows.append({**ok, "query": it["query"],
                     "missing_ok": pred.missing_fields() == TravelRequest(**it["gold"]).missing_fields(),
                     "pred": pred.model_dump_json(exclude_defaults=True)})
        print(("OK  " if all(ok.values()) else "ERR ") + it["query"])
    df = pd.DataFrame(rows)
    fields = list(TravelRequest.model_fields)
    res = {"model": llm.model, "n": len(df),
           "exact_match": round(df[fields].all(axis=1).mean(), 3),
           "clarification_detection": round(df["missing_ok"].mean(), 3),
           "field_accuracy": df[fields].mean().round(3).to_dict(),
           "latency_s_median": round(float(pd.Series(latencies).median()), 2)}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "parser_metrics.json").write_text(json.dumps(res, indent=2))
    df[~df[fields].all(axis=1)].to_csv(ROOT / "reports" / "parser_errors.csv", index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
