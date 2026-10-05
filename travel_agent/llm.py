"""Client LLM minimal, compatible OpenAI et Mistral (API compatible OpenAI).

Le client expose une seule méthode utile aux agents : `chat_json`, qui force
une réponse JSON et la valide avec un modèle Pydantic. En cas de JSON invalide,
on relance avec le message d'erreur (auto-correction), jusqu'à `max_retries`.
"""
from __future__ import annotations

import json
import os
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

PROVIDERS = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY",
                "default_model": "mistral-small-latest"},
}


class LLM(Protocol):
    def chat_json(self, system: str, user: str, schema: type[T]) -> T: ...


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float = 0.0, max_retries: int = 2):
        from openai import OpenAI  # import local : les tests n'en ont pas besoin

        provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        if provider not in PROVIDERS:
            raise ValueError(f"Fournisseur inconnu : {provider} (attendu : {list(PROVIDERS)})")
        cfg = PROVIDERS[provider]
        api_key = os.getenv(cfg["key_env"])
        if not api_key:
            raise RuntimeError(f"Variable d'environnement {cfg['key_env']} manquante (voir .env.example)")
        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        self.model = model or os.getenv("LLM_MODEL", cfg["default_model"])
        self.temperature = temperature
        self.max_retries = max_retries
        self.calls = 0

    def chat_json(self, system: str, user: str, schema: type[T]) -> T:
        messages = [
            {"role": "system", "content": system + "\n\nRéponds UNIQUEMENT avec un objet JSON valide "
             f"respectant ce schéma :\n{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"},
            {"role": "user", "content": user},
        ]
        last_error = None
        for _ in range(self.max_retries + 1):
            self.calls += 1
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(content)
            except ValidationError as err:
                last_error = err
                messages += [{"role": "assistant", "content": content},
                             {"role": "user", "content": f"JSON invalide : {err}. Corrige ta réponse."}]
        raise RuntimeError(f"Réponse LLM invalide après {self.max_retries + 1} essais : {last_error}")
