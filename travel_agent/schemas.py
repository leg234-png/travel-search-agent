"""Schémas Pydantic : requête structurée extraite par le LLM et réponses de l'agent."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Preference = Literal["cheap", "fast", "comfort", "balanced"]
TimeWindow = Literal["morning", "afternoon", "evening", "any"]


class TravelRequest(BaseModel):
    origin: str | None = Field(None, description="Code ville d'origine (ex : PAR), null si absent")
    destination: str | None = Field(None, description="Code ville de destination, null si absent")
    date_from: date | None = Field(None, description="Première date de départ acceptable (AAAA-MM-JJ)")
    date_to: date | None = Field(None, description="Dernière date de départ acceptable (AAAA-MM-JJ)")
    max_price: float | None = Field(None, description="Budget maximum du vol en euros")
    max_stops: int | None = Field(None, description="Nombre maximum d'escales (0 = direct)")
    preference: Preference = Field("balanced", description="cheap, fast, comfort ou balanced")
    time_window: TimeWindow = Field("any", description="Créneau de départ souhaité")
    nights: int | None = Field(None, description="Nombre de nuits d'hôtel, null si pas d'hôtel")
    hotel_max_price: float | None = Field(None, description="Prix max par nuit")
    hotel_min_stars: int | None = None
    hotel_near_center: bool = False

    def missing_fields(self) -> list[str]:
        return [f for f in ("origin", "destination", "date_from") if getattr(self, f) is None]


class Recommendation(BaseModel):
    flight_ids: list[int] = Field(description="Identifiants des vols recommandés, du meilleur au moins bon")
    hotel_ids: list[int] = Field(default_factory=list)
    message: str = Field(description="Réponse à l'utilisateur (3 à 6 phrases), en citant prix et horaires")


class AgentAnswer(BaseModel):
    status: Literal["ok", "clarification", "no_result"]
    message: str
    request: TravelRequest | None = None
    flights: list[dict] = Field(default_factory=list)
    hotels: list[dict] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
