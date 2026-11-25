from typing import Any, Protocol

import httpx

from fm_scout.schema import ScoutError

Message = dict[str, Any]


class LLMClient(Protocol):
    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        """Renvoie le message « assistant » au format OpenAI."""
        ...


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, timeout: float, reasoning_effort: str = ""):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = timeout
        self._effort = reasoning_effort

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        body: dict[str, Any] = {"model": self._model, "messages": messages, "tools": tools,
                                "temperature": 0, "seed": 1, "stream": False}
        if self._effort:
            body["reasoning_effort"] = self._effort
        try:
            response = httpx.post(self._url, json=body, timeout=self._timeout)
        except httpx.TimeoutException as e:
            raise ScoutError(f"Le modèle n'a pas répondu en {self._timeout:.0f} s.") from e
        except httpx.HTTPError as e:
            raise ScoutError(f"Serveur LLM injoignable ({self._url}). Ollama est-il lancé ? "
                             "Sinon : --no-llm.") from e
        if response.status_code != 200:
            raise ScoutError(f"Erreur du serveur LLM ({response.status_code}, modèle "
                             f"« {self._model} ») : {response.text[:300]}")
        try:
            return response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as e:
            raise ScoutError("Réponse illisible du serveur LLM.") from e
