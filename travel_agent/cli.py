"""Interroger l'agent en langage naturel.

    python -m travel_agent.cli "Un vol direct Paris-Lisbonne fin mai, le moins cher possible, avec 3 nuits d'hôtel près du centre"
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

from .agent import TravelAgent, _fmt_hour
from .llm import LLMClient


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    load_dotenv()
    ans = TravelAgent(LLMClient()).run(" ".join(sys.argv[1:]))
    print(f"\n[{ans.status}] {ans.message}\n")
    for f in ans.flights:
        print(f"  vol #{f['id']}: {f['depart_date']} {_fmt_hour(f['depart_hour'])}  {f['airline']:10s} "
              f"{f['stops']} escale(s)  {f['duration_min']} min  {f['price_eur']} €")
    for h in ans.hotels:
        print(f"  hôtel #{h['id']}: {h['name']} {h['stars']}*  note {h['rating']}  {h['price_night_eur']} €/nuit")
    print("\nTrace :", *ans.trace, sep="\n  - ")


if __name__ == "__main__":
    main()
