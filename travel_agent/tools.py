"""Outils de recherche appelés par l'agent : requêtes SQL paramétrées sur le catalogue."""
from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .catalog import DB_PATH
from .schemas import TravelRequest

WINDOWS = {"morning": (5, 12), "afternoon": (12, 18), "evening": (18, 24), "any": (0, 24)}


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} introuvable : lancez `python -m travel_agent.catalog`")
    return sqlite3.connect(db_path)


def city_codes(con: sqlite3.Connection) -> dict[str, str]:
    return dict(con.execute("SELECT code, name FROM cities").fetchall())


def search_flights(con: sqlite3.Connection, req: TravelRequest, limit: int = 200) -> pd.DataFrame:
    """Filtre dur (contraintes explicites) ; le classement est délégué au ranker."""
    date_to = req.date_to or req.date_from + timedelta(days=0)
    sql = ["SELECT * FROM flights WHERE origin = ? AND destination = ? AND depart_date BETWEEN ? AND ?"]
    params: list = [req.origin, req.destination, req.date_from.isoformat(), date_to.isoformat()]
    if req.max_price is not None:
        sql.append("AND price_eur <= ?")
        params.append(req.max_price)
    if req.max_stops is not None:
        sql.append("AND stops <= ?")
        params.append(req.max_stops)
    sql.append("LIMIT ?")
    params.append(limit)
    return pd.read_sql_query(" ".join(sql), con, params=params)


def search_hotels(con: sqlite3.Connection, req: TravelRequest, k: int = 5) -> pd.DataFrame:
    sql = ["SELECT * FROM hotels WHERE city = ?"]
    params: list = [req.destination]
    if req.hotel_max_price is not None:
        sql.append("AND price_night_eur <= ?")
        params.append(req.hotel_max_price)
    if req.hotel_min_stars is not None:
        sql.append("AND stars >= ?")
        params.append(req.hotel_min_stars)
    if req.hotel_near_center:
        sql.append("AND dist_center_km <= 2.0")
    # Score simple et lisible : note client, pénalité prix et distance
    sql.append("ORDER BY rating - 0.01 * price_night_eur - 0.3 * dist_center_km DESC LIMIT ?")
    params.append(k)
    df = pd.read_sql_query(" ".join(sql), con, params=params)
    if req.nights:
        df["total_eur"] = (df["price_night_eur"] * req.nights).round(2)
    return df
