import httpx
from pydantic import BaseModel, Field


class Chat(BaseModel):
    """Classe de base pour une requête POST générique vers un LLM.
    * Le modèle à utiliser.
    * La température.
    * Les messages envoyés.
    """

    model: str
    # Utilisation de Field pour valider les ressources (CPU/GPU) via la température
    temperature: float | None = Field(ge=0.0, le=1.0, default=0.7)
    messages: list[dict[str, str]]


class LLMClient:
    """Le client utilisé pour communiquer avec le LLM en local."""

    def __init__(self, root_url: str) -> None:
        # Pour une installation complètement locale, verify=True est standard
        self.client = httpx.Client(verify=True)
        self.root_url = root_url

    def _generate_request(self, chat: Chat) -> tuple[dict, dict, str]:
        """Génère les 3 parties nécessaires pour la requête via HTTPX."""
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        # Structure du corps de la requête pour Ollama
        body = {
            "model": chat.model,
            "messages": chat.messages,
            "stream": False,
            "options": {"temperature": chat.temperature},
        }
        # Construction de l'URL locale
        route = f"http://{self.root_url}/api/chat"
        return headers, body, route

    def post(self, chat: Chat):
        """Envoie la requête POST au serveur Ollama."""
        headers, body, route = self._generate_request(chat=chat)
        try:
            response = self.client.post(
                url=route,
                headers=headers,
                json=body,
                timeout=180.0,  # Timeout long car les LLM peuvent être lents sur CPU
            )
            response.raise_for_status()
            return response
        except httpx.RequestError as exc:
            print(f"Une erreur est survenue lors de la requête : {exc.request.url!r}.")
            raise
        except httpx.HTTPStatusError as exc:
            print(f"Erreur status {exc.response.status_code} pour {exc.request.url!r}.")
            raise


# Instance du client pointant vers l'installation locale par défaut
client = LLMClient(root_url="localhost:11434")
