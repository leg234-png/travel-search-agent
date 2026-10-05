"""Construction d'un catalogue de voyages synthétique dans SQLite (vols + hôtels).

Les prix, durées et escales suivent des règles réalistes (distance, heure de départ,
nombre d'escales, compagnie) avec du bruit. Graine fixe : catalogue reproductible.

    python -m travel_agent.catalog          # écrit data/travel.db
"""
from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "travel.db"

CITIES = [  # code, nom, pays, lat, lon
    ("PAR", "Paris", "France", 48.86, 2.35), ("LYS", "Lyon", "France", 45.76, 4.84),
    ("NCE", "Nice", "France", 43.70, 7.27), ("MRS", "Marseille", "France", 43.30, 5.37),
    ("TLS", "Toulouse", "France", 43.60, 1.44), ("BOD", "Bordeaux", "France", 44.84, -0.58),
    ("BES", "Brest", "France", 48.39, -4.49), ("LIS", "Lisbonne", "Portugal", 38.72, -9.14),
    ("MAD", "Madrid", "Espagne", 40.42, -3.70), ("BCN", "Barcelone", "Espagne", 41.39, 2.17),
    ("ROM", "Rome", "Italie", 41.90, 12.50), ("MIL", "Milan", "Italie", 45.46, 9.19),
    ("BER", "Berlin", "Allemagne", 52.52, 13.40), ("AMS", "Amsterdam", "Pays-Bas", 52.37, 4.90),
    ("LON", "Londres", "Royaume-Uni", 51.51, -0.13), ("BRU", "Bruxelles", "Belgique", 50.85, 4.35),
    ("DUB", "Dublin", "Irlande", 53.35, -6.26), ("PRG", "Prague", "Tchéquie", 50.08, 14.44),
    ("VIE", "Vienne", "Autriche", 48.21, 16.37), ("ATH", "Athènes", "Grèce", 37.98, 23.73),
]
# compagnie : (multiplicateur de prix, note qualité /5)
AIRLINES = {"AirFrance": (1.35, 4.2), "Lufthansa": (1.30, 4.1), "Transavia": (0.85, 3.4),
            "EasyJet": (0.80, 3.3), "Ryanair": (0.65, 2.8), "Vueling": (0.82, 3.2), "TAP": (1.05, 3.6)}
START, N_DAYS = date(2027, 4, 1), 61
HOTEL_PREFIXES = ["Hôtel", "Grand Hôtel", "Résidence", "Auberge", "Boutique Hôtel", "Ibis", "Novotel"]
HOTEL_NAMES = ["du Centre", "de la Gare", "des Arts", "Le Méridien", "Bellevue", "du Port", "Royal",
               "Les Jardins", "Panorama", "Saint-Michel", "Opéra", "Riviera", "Lumière"]

SCHEMA = """
CREATE TABLE cities (code TEXT PRIMARY KEY, name TEXT, country TEXT, lat REAL, lon REAL);
CREATE TABLE flights (
    id INTEGER PRIMARY KEY, origin TEXT, destination TEXT, depart_date TEXT, depart_hour REAL,
    duration_min INTEGER, stops INTEGER, airline TEXT, airline_rating REAL, price_eur REAL);
CREATE INDEX idx_flights ON flights(origin, destination, depart_date);
CREATE TABLE hotels (
    id INTEGER PRIMARY KEY, city TEXT, name TEXT, stars INTEGER, rating REAL,
    price_night_eur REAL, dist_center_km REAL, breakfast INTEGER);
CREATE INDEX idx_hotels ON hotels(city);
"""


def haversine_km(a: tuple, b: tuple) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[3], a[4], b[3], b[4]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def build(db_path: Path = DB_PATH, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO cities VALUES (?,?,?,?,?)", CITIES)

    flights, fid = [], 0
    airlines = list(AIRLINES)
    for o in CITIES:
        for d in CITIES:
            if o == d:
                continue
            dist = haversine_km(o, d)
            for day in range(N_DAYS):
                dep_date = (START + timedelta(days=day)).isoformat()
                weekend = (START + timedelta(days=day)).weekday() >= 4
                for _ in range(rng.integers(2, 6)):
                    stops = int(rng.choice([0, 0, 0, 1, 1, 2])) if dist > 600 else int(rng.choice([0, 0, 1]))
                    airline = airlines[rng.integers(len(airlines))]
                    mult, rating = AIRLINES[airline]
                    hour = float(np.round(rng.uniform(6, 22) * 4) / 4)
                    duration = dist / 780 * 60 + 40 + stops * rng.uniform(70, 180) + rng.normal(0, 10)
                    peak = 1.25 if (7 <= hour <= 9 or 17 <= hour <= 19) else 1.0
                    price = (35 + 0.11 * dist) * mult * peak * (1.15 if weekend else 1.0)
                    price *= (0.78 ** stops) * rng.lognormal(0, 0.18)
                    fid += 1
                    flights.append((fid, o[0], d[0], dep_date, hour, int(duration), stops, airline,
                                    rating, round(float(price), 2)))
    con.executemany("INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?)", flights)

    hotels, hid = [], 0
    for c in CITIES:
        for _ in range(25):
            stars = int(rng.choice([1, 2, 3, 3, 3, 4, 4, 5]))
            dist_center = float(np.round(rng.gamma(2.0, 1.2), 1))
            rating = float(np.clip(5.5 + stars * 0.7 + rng.normal(0, 0.8) - dist_center * 0.1, 3, 10))
            price = 30 + stars ** 1.6 * 18 * rng.lognormal(0, 0.2) * (1.2 if c[0] in ("PAR", "LON", "AMS") else 1)
            price *= max(0.6, 1 - dist_center * 0.04)
            hid += 1
            hotels.append((hid, c[0], f"{rng.choice(HOTEL_PREFIXES)} {rng.choice(HOTEL_NAMES)}", stars,
                           round(rating, 1), round(float(price), 2), dist_center, int(rng.random() < 0.5)))
    con.executemany("INSERT INTO hotels VALUES (?,?,?,?,?,?,?,?)", hotels)
    con.commit()
    con.close()
    print(f"Catalogue écrit dans {db_path} : {len(flights)} vols, {len(hotels)} hôtels")
    return db_path


if __name__ == "__main__":
    build()
