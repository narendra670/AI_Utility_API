import httpx

from app.utils.logger import logger
from config import get_settings

settings = get_settings()


class AIProviderError(Exception):
    """Raised when the upstream AI provider returns an error."""


class AIService:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ai_base_url).rstrip("/")
        self.api_key = api_key or settings.ai_api_key
        self.model = model or settings.ai_model
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=settings.ai_timeout,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not self.api_key:
            raise AIProviderError(
                "AI_API_KEY is not configured. Set it in .env or the environment."
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        logger.debug("Sending chat request: %s", payload)
        response = self._client.post("/chat/completions", json=payload)

        if response.status_code != 200:
            logger.error(
                "AI provider error %s: %s", response.status_code, response.text
            )
            raise AIProviderError(
                f"AI provider returned HTTP {response.status_code}"
            )

        content = response.json()["choices"][0]["message"]["content"].strip()
        logger.debug("AI response: %s", content)
        return content

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._chat(
            messages,
            max_tokens or settings.ai_max_tokens,
            temperature if temperature is not None else settings.ai_temperature,
        )

    def summarize(self, text: str, max_words: int = 100) -> str:
        return self.generate(
            f"Summarize the following text in at most {max_words} words:\n\n{text}",
            system="You are a concise summarizer.",
            temperature=0.3,
        )

    def grammar_check(self, text: str) -> str:
        return self.generate(
            f"Fix the grammar and spelling of the following text and return only the corrected version:\n\n{text}",
            system="You are a precise grammar and style editor.",
            temperature=0.2,
        )

    def close(self) -> None:
        self._client.close()
