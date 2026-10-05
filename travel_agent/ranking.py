"""Ranking des vols : features, utilisateurs simulés, baselines, modèles appris et métriques.

Pertinence : on ne dispose pas de clics réels. On simule des voyageurs dont l'utilité dépend
de leur préférence déclarée (cheap / fast / comfort / balanced) et de poids personnels cachés
(bruit log-normal). La pertinence graduée (0 à 3) est déduite du rang d'utilité dans chaque requête.
Les modèles ne voient que des features observables, jamais les poids cachés.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .schemas import TravelRequest
from .tools import WINDOWS

PREFS = ["cheap", "fast", "comfort", "balanced"]
FEATURES = ["log_price_rel", "price_pct", "log_dur_rel", "dur_pct", "stops", "airline_rating",
            "window_gap_h", "dep_hour", "n_candidates"] + [f"pref_{p}" for p in PREFS]

# Poids moyens de l'utilité simulée : (prix, durée, escales, qualité compagnie, créneau)
TRUE_WEIGHTS = {
    "cheap": (3.0, 0.7, 0.4, 0.2, 0.6),
    "fast": (0.8, 3.0, 1.2, 0.3, 0.6),
    "comfort": (0.8, 1.0, 1.5, 1.4, 0.6),
    "balanced": (1.5, 1.3, 0.8, 0.5, 0.6),
}


# ---------------------------------------------------------------- features
def window_gap(hours: pd.Series, window: str) -> pd.Series:
    lo, hi = WINDOWS[window]
    return np.maximum(0, np.maximum(lo - hours, hours - hi))


def build_features(cands: pd.DataFrame, req: TravelRequest) -> pd.DataFrame:
    """Features relatives à la requête (prix rapporté au moins cher de la liste, etc.)."""
    f = pd.DataFrame(index=cands.index)
    f["log_price_rel"] = np.log(cands["price_eur"] / cands["price_eur"].min())
    f["price_pct"] = cands["price_eur"].rank(pct=True)
    f["log_dur_rel"] = np.log(cands["duration_min"] / cands["duration_min"].min())
    f["dur_pct"] = cands["duration_min"].rank(pct=True)
    f["stops"] = cands["stops"]
    f["airline_rating"] = cands["airline_rating"]
    f["window_gap_h"] = window_gap(cands["depart_hour"], req.time_window)
    f["dep_hour"] = cands["depart_hour"]
    f["n_candidates"] = len(cands)
    for p in PREFS:
        f[f"pref_{p}"] = float(req.preference == p)
    return f[FEATURES]


# ------------------------------------------------------ utilisateurs simulés
def simulate_relevance(feats: pd.DataFrame, pref: str, rng: np.random.Generator) -> np.ndarray:
    w = np.array(TRUE_WEIGHTS[pref]) * rng.lognormal(0, 0.35, size=5)  # préférences personnelles
    utility = (-w[0] * feats["log_price_rel"] - w[1] * feats["log_dur_rel"] - w[2] * feats["stops"]
               + w[3] * (feats["airline_rating"] - 3.5) - w[4] * feats["window_gap_h"] / 3
               + rng.gumbel(0, 0.25, size=len(feats)))
    order = (-utility).argsort().argsort()  # rang 0 = meilleur
    return np.select([order == 0, order < 3, order < 8], [3, 2, 1], default=0)


# --------------------------------------------------------------- rankers
class PriceBaseline:
    name = "baseline_prix"

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        return -feats["log_price_rel"].to_numpy()


class HeuristicBaseline:
    """Ce qu'un développeur écrirait à la main : poids fixes selon la préférence déclarée."""
    name = "baseline_heuristique"
    W = {"cheap": (2, 0.5, 0.5, 0.2), "fast": (0.5, 2, 1, 0.2), "comfort": (0.5, 0.5, 1, 1),
         "balanced": (1, 1, 1, 0.3)}

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        pref = next(p for p in PREFS if feats[f"pref_{p}"].iloc[0] == 1)
        a, b, c, d = self.W[pref]
        return (-a * feats["price_pct"] - b * feats["dur_pct"] - c * feats["stops"]
                + d * feats["airline_rating"] - 0.3 * feats["window_gap_h"]).to_numpy()


class PointwiseGBM:
    """Régression de la pertinence graduée (gradient boosting, interactions apprises)."""
    name = "gbm_pointwise"

    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5,
                                                   l2_regularization=1.0, random_state=seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> "PointwiseGBM":
        self.model.fit(X, y)
        return self

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        return self.model.predict(feats)


class PairwiseLogReg:
    """RankNet linéaire : régression logistique sur des différences de features (paires intra-requête)."""
    name = "logreg_pairwise"

    def __init__(self, pairs_per_query: int = 60, seed: int = 0):
        self.k, self.rng = pairs_per_query, np.random.default_rng(seed)
        self.model = LogisticRegression(max_iter=2000, fit_intercept=False)

    def _expand(self, X: pd.DataFrame) -> np.ndarray:
        # interactions préférence × attributs, pour qu'un modèle linéaire capte les préférences
        base = X[FEATURES[:7]].to_numpy()
        prefs = X[[f"pref_{p}" for p in PREFS]].to_numpy()
        return np.hstack([base] + [base * prefs[:, [j]] for j in range(len(PREFS))])

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> "PairwiseLogReg":
        Z = self._expand(X)
        diffs, labels = [], []
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            for _ in range(self.k):
                i, j = self.rng.choice(idx, 2, replace=False)
                if y[i] == y[j]:
                    continue
                diffs.append(Z[i] - Z[j])
                labels.append(int(y[i] > y[j]))
        self.model.fit(np.array(diffs), np.array(labels))
        return self

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        return self._expand(feats) @ self.model.coef_.ravel()


# --------------------------------------------------------------- métriques
def dcg(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    return float(np.sum((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    ideal = dcg(np.sort(y_true)[::-1], k)
    return dcg(y_true[np.argsort(-scores, kind="stable")], k) / ideal if ideal > 0 else 0.0


def mrr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rang réciproque de la meilleure option (pertinence 3)."""
    ranked = y_true[np.argsort(-scores, kind="stable")]
    hits = np.where(ranked == 3)[0]
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    return float(np.mean(y_true[np.argsort(-scores, kind="stable")][:k] >= 2))
