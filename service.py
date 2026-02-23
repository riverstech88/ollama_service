from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Optional

import requests

from ollama_service_updated.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    ModelDetail,
    ModelInfo,
    PullRequest,
    PullResponse,
)

logger = logging.getLogger("ollama_service")


class OllamaServiceError(Exception):
    pass


class OllamaConnectionError(OllamaServiceError):
    pass


class OllamaAPIError(OllamaServiceError):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class OllamaAuthError(OllamaServiceError):
    pass


class OllamaService:
    def __init__(
        self,
        host: str = "18.205.235.191",
        port: int = 11434,
        timeout: int = 120,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._session = requests.Session()
        self._api_key = api_key
        self._auth = None

        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
            logger.info("Configured Bearer token authentication")

        if username and password:
            self._auth = (username, password)
            self._session.auth = self._auth
            logger.info("Configured basic authentication for user=%s", username)

        if headers:
            self._session.headers.update(headers)
            logger.info("Configured custom headers: %s", list(headers.keys()))

        logger.info("Initialized OllamaService at %s", self.base_url)

    @property
    def authenticated(self) -> bool:
        return self._api_key is not None or self._auth is not None

    def verify_auth(self) -> bool:
        url = f"{self.base_url}/api/tags"
        logger.info("REQUEST  | method=GET | url=%s | payload_keys=[] | authenticated=%s | action=verify_auth", url, self.authenticated)
        start = time.monotonic()
        try:
            response = self._session.get(url, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            elapsed = time.monotonic() - start
            logger.error(
                "FAILURE  | method=GET | url=%s | error=%s | elapsed=%.3fs | action=verify_auth | detail=%s",
                url, type(e).__name__.lower(), elapsed, e,
            )
            raise OllamaConnectionError(f"Failed to connect to {self.base_url}: {e}")
        elapsed = time.monotonic() - start
        logger.info(
            "RESPONSE | method=GET | url=%s | status=%d | elapsed=%.3fs | size=%d bytes | action=verify_auth",
            url, response.status_code, elapsed, len(response.content),
        )
        if response.status_code in (401, 403):
            logger.error(
                "AUTH_ERR | method=GET | url=%s | status=%d | elapsed=%.3fs | action=verify_auth",
                url, response.status_code, elapsed,
            )
            return False
        return response.status_code == 200

    def _request(self, method: str, endpoint: str, json_data: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        payload_keys = list(json_data.keys()) if json_data else []
        logger.info(
            "REQUEST  | method=%s | url=%s | payload_keys=%s | authenticated=%s",
            method, url, payload_keys, self.authenticated,
        )
        start = time.monotonic()
        try:
            response = self._session.request(
                method=method,
                url=url,
                json=json_data,
                timeout=self.timeout,
            )
        except requests.ConnectionError as e:
            elapsed = time.monotonic() - start
            logger.error(
                "FAILURE  | method=%s | url=%s | error=connection_error | elapsed=%.3fs | detail=%s",
                method, url, elapsed, e,
            )
            raise OllamaConnectionError(f"Failed to connect to {self.base_url}: {e}")
        except requests.Timeout as e:
            elapsed = time.monotonic() - start
            logger.error(
                "FAILURE  | method=%s | url=%s | error=timeout | elapsed=%.3fs | timeout_limit=%ds | detail=%s",
                method, url, elapsed, self.timeout, e,
            )
            raise OllamaConnectionError(f"Request timed out: {e}")

        elapsed = time.monotonic() - start
        response_size = len(response.content)
        logger.info(
            "RESPONSE | method=%s | url=%s | status=%d | elapsed=%.3fs | size=%d bytes",
            method, url, response.status_code, elapsed, response_size,
        )

        if response.status_code in (401, 403):
            logger.error(
                "AUTH_ERR | method=%s | url=%s | status=%d | elapsed=%.3fs | body=%s",
                method, url, response.status_code, elapsed, response.text[:500],
            )
            raise OllamaAuthError(
                f"Authentication failed with status {response.status_code}: {response.text}"
            )

        if response.status_code != 200:
            logger.error(
                "API_ERR  | method=%s | url=%s | status=%d | elapsed=%.3fs | body=%s",
                method, url, response.status_code, elapsed, response.text[:500],
            )
            raise OllamaAPIError(response.status_code, response.text)

        return response.json()

    def is_healthy(self) -> bool:
        url = f"{self.base_url}/"
        logger.info("REQUEST  | method=GET | url=%s | payload_keys=[] | authenticated=%s", url, self.authenticated)
        start = time.monotonic()
        try:
            response = self._session.get(url, timeout=self.timeout)
            elapsed = time.monotonic() - start
            healthy = response.status_code == 200
            logger.info(
                "RESPONSE | method=GET | url=%s | status=%d | elapsed=%.3fs | size=%d bytes | healthy=%s",
                url, response.status_code, elapsed, len(response.content), healthy,
            )
            return healthy
        except requests.ConnectionError as e:
            elapsed = time.monotonic() - start
            logger.warning(
                "FAILURE  | method=GET | url=%s | error=connection_error | elapsed=%.3fs | detail=%s",
                url, elapsed, e,
            )
            return False
        except requests.Timeout as e:
            elapsed = time.monotonic() - start
            logger.warning(
                "FAILURE  | method=GET | url=%s | error=timeout | elapsed=%.3fs | timeout_limit=%ds | detail=%s",
                url, elapsed, self.timeout, e,
            )
            return False

    def list_models(self) -> list[ModelInfo]:
        data = self._request("GET", "/api/tags")
        models = []
        for m in data.get("models", []):
            models.append(
                ModelInfo(
                    name=m.get("name", ""),
                    model=m.get("model", ""),
                    size=m.get("size", 0),
                    digest=m.get("digest", ""),
                    modified_at=m.get("modified_at", ""),
                    details=m.get("details", {}),
                )
            )
        model_names = [m.name for m in models]
        logger.info("RESULT   | action=list_models | count=%d | models=%s", len(models), model_names)
        return models

    def list_model_names(self) -> list[str]:
        return [m.name for m in self.list_models()]

    def show_model(self, model_name: str) -> ModelDetail:
        data = self._request("POST", "/api/show", {"name": model_name})
        return ModelDetail(
            modelfile=data.get("modelfile", ""),
            parameters=data.get("parameters", ""),
            template=data.get("template", ""),
            details=data.get("details", {}),
            model_info=data.get("model_info", {}),
        )

    def pull_model(self, model_name: str) -> PullResponse:
        req = PullRequest(name=model_name, stream=False)
        data = self._request("POST", "/api/pull", asdict(req))
        status = data.get("status", "")
        logger.info("RESULT   | action=pull_model | model=%s | status=%s", model_name, status)
        return PullResponse(status=status)

    def delete_model(self, model_name: str) -> bool:
        url = f"{self.base_url}/api/delete"
        logger.info(
            "REQUEST  | method=DELETE | url=%s | payload_keys=['name'] | authenticated=%s | model=%s",
            url, self.authenticated, model_name,
        )
        start = time.monotonic()
        try:
            response = self._session.delete(url, json={"name": model_name}, timeout=self.timeout)
        except requests.ConnectionError as e:
            elapsed = time.monotonic() - start
            logger.error(
                "FAILURE  | method=DELETE | url=%s | error=connection_error | elapsed=%.3fs | detail=%s",
                url, elapsed, e,
            )
            raise OllamaConnectionError(f"Failed to connect to {self.base_url}: {e}")
        elapsed = time.monotonic() - start
        success = response.status_code == 200
        logger.info(
            "RESPONSE | method=DELETE | url=%s | status=%d | elapsed=%.3fs | size=%d bytes | success=%s",
            url, response.status_code, elapsed, len(response.content), success,
        )
        return success

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload = asdict(request)
        payload = {k: v for k, v in payload.items() if v is not None}
        data = self._request("POST", "/api/generate", payload)
        response = GenerateResponse(
            model=data.get("model", ""),
            response=data.get("response", ""),
            done=data.get("done", False),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
            context=data.get("context"),
        )
        logger.info(
            "RESULT   | action=generate | model=%s | done=%s | prompt_eval_count=%s | eval_count=%s | total_duration_ns=%s | load_duration_ns=%s | eval_duration_ns=%s | response_length=%d",
            response.model, response.done, response.prompt_eval_count,
            response.eval_count, response.total_duration, response.load_duration,
            response.eval_duration, len(response.response),
        )
        return response

    def generate_simple(self, model: str, prompt: str, system: Optional[str] = None) -> str:
        req = GenerateRequest(model=model, prompt=prompt, system=system, stream=False)
        return self.generate(req).response

    def chat(self, request: ChatRequest) -> ChatResponse:
        payload = asdict(request)
        payload = {k: v for k, v in payload.items() if v is not None}
        data = self._request("POST", "/api/chat", payload)
        msg = data.get("message", {})
        response = ChatResponse(
            model=data.get("model", ""),
            message=ChatMessage(role=msg.get("role", ""), content=msg.get("content", "")),
            done=data.get("done", False),
            total_duration=data.get("total_duration"),
            load_duration=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
        )
        logger.info(
            "RESULT   | action=chat | model=%s | messages=%d | done=%s | role=%s | prompt_eval_count=%s | eval_count=%s | total_duration_ns=%s | load_duration_ns=%s | eval_duration_ns=%s | response_length=%d",
            response.model, len(request.messages), response.done, response.message.role,
            response.prompt_eval_count, response.eval_count, response.total_duration,
            response.load_duration, response.eval_duration, len(response.message.content),
        )
        return response

    def chat_simple(self, model: str, user_message: str, system_message: Optional[str] = None) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        req = ChatRequest(model=model, messages=messages, stream=False)
        return self.chat(req).message.content

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        payload = asdict(request)
        data = self._request("POST", "/api/embed", payload)
        response = EmbeddingResponse(
            model=data.get("model", ""),
            embeddings=data.get("embeddings", []),
        )
        dimensions = len(response.embeddings[0]) if response.embeddings else 0
        logger.info(
            "RESULT   | action=embeddings | model=%s | vectors=%d | dimensions=%d",
            response.model, len(response.embeddings), dimensions,
        )
        return response

    def get_embedding(self, model: str, text: str) -> list[float]:
        req = EmbeddingRequest(model=model, input=text)
        resp = self.embeddings(req)
        if resp.embeddings:
            return resp.embeddings[0]
        return []

    def list_running(self) -> list[dict]:
        data = self._request("GET", "/api/ps")
        models = data.get("models", [])
        logger.info("RESULT   | action=list_running | count=%d", len(models))
        return models

    def close(self):
        logger.info("Closing OllamaService session")
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
