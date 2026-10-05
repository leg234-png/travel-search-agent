"""Entraîne et évalue les rankers de vols avec une validation croisée groupée par requête.

    python -m travel_agent.train_ranker --queries 3000

Le split est fait par requête (GroupKFold) : toutes les options d'une même recherche sont soit
dans le train soit dans le test. Sinon, les features relatives à la requête feraient fuiter
l'information entre train et test.
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .catalog import CITIES, N_DAYS, START
from .ranking import (FEATURES, PREFS, HeuristicBaseline, PairwiseLogReg, PointwiseGBM, PriceBaseline,
                      build_features, mrr, ndcg_at_k, precision_at_k, simulate_relevance)
from .schemas import TravelRequest
from .tools import connect, search_flights

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ranker.joblib"
REPORTS = ROOT / "reports"


def sample_requests(n: int, rng: np.random.Generator) -> list[TravelRequest]:
    codes = [c[0] for c in CITIES]
    reqs = []
    for _ in range(n):
        o, d = rng.choice(codes, 2, replace=False)
        start = START + timedelta(days=int(rng.integers(0, N_DAYS - 4)))
        reqs.append(TravelRequest(
            origin=o, destination=d, date_from=start, date_to=start + timedelta(days=int(rng.integers(0, 4))),
            max_stops=rng.choice([None, None, 1, 0]), preference=rng.choice(PREFS),
            time_window=rng.choice(["any", "any", "morning", "afternoon", "evening"])))
    return reqs


def build_dataset(n_queries: int, seed: int = 0, max_cands: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    con = connect()
    frames = []
    for qid, req in enumerate(sample_requests(n_queries, rng)):
        cands = search_flights(con, req)
        if len(cands) < 10:
            continue
        if len(cands) > max_cands:
            cands = cands.sample(max_cands, random_state=int(rng.integers(1e9)))
        feats = build_features(cands, req)
        feats["relevance"] = simulate_relevance(feats, req.preference, rng)
        feats["qid"] = qid
        feats["preference"] = req.preference
        frames.append(feats.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def evaluate(rankers: dict, df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    rows = []
    groups = df["qid"].to_numpy()
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=groups)):
        train, test = df.iloc[tr], df.iloc[te]
        for name, make in rankers.items():
            model = make()
            if hasattr(model, "fit"):
                model.fit(train[FEATURES], train["relevance"].to_numpy(), train["qid"].to_numpy())
            for qid, q in test.groupby("qid"):
                s = model.score(q[FEATURES])
                y = q["relevance"].to_numpy()
                rows.append({"fold": fold, "ranker": name, "preference": q["preference"].iloc[0],
                             "ndcg@5": ndcg_at_k(y, s, 5), "mrr": mrr(y, s), "p@5": precision_at_k(y, s, 5)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = build_dataset(args.queries, args.seed)
    print(f"{df['qid'].nunique()} requêtes, {len(df)} options candidates")
    rankers = {"baseline_prix": PriceBaseline, "baseline_heuristique": HeuristicBaseline,
               "logreg_pairwise": lambda: PairwiseLogReg(seed=args.seed),
               "gbm_pointwise": lambda: PointwiseGBM(seed=args.seed)}
    res = evaluate(rankers, df)

    per_fold = res.groupby(["ranker", "fold"])[["ndcg@5", "mrr", "p@5"]].mean()
    summary = per_fold.groupby("ranker").agg(["mean", "std"]).round(3)
    by_pref = res.groupby(["ranker", "preference"])["ndcg@5"].mean().unstack().round(3)
    order = ["baseline_prix", "baseline_heuristique", "logreg_pairwise", "gbm_pointwise"]
    summary, by_pref = summary.loc[order], by_pref.loc[order]
    print("\nValidation croisée groupée (5 folds) :\n", summary)
    print("\nNDCG@5 par préférence déclarée :\n", by_pref)

    REPORTS.mkdir(exist_ok=True)
    out = {"n_queries": int(df["qid"].nunique()), "n_candidates": len(df),
           "summary": {r: {m: {"mean": float(summary.loc[r, (m, "mean")]), "std": float(summary.loc[r, (m, "std")])}
                           for m in ["ndcg@5", "mrr", "p@5"]} for r in order},
           "ndcg@5_by_preference": by_pref.to_dict(orient="index")}
    (REPORTS / "ranking_metrics.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    by_pref.T.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("NDCG@5")
    ax.set_title("NDCG@5 par préférence et par ranker (CV groupée)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "ndcg_by_preference.png", dpi=120)

    # Modèle final : le meilleur modèle appris (NDCG@5 moyen en CV), réentraîné sur toutes les requêtes
    learned = ["logreg_pairwise", "gbm_pointwise"]
    best = max(learned, key=lambda r: summary.loc[r, ("ndcg@5", "mean")])
    final = rankers[best]().fit(df[FEATURES], df["relevance"].to_numpy(), df["qid"].to_numpy())
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"\nModèle final ({best}) sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
