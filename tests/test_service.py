import logging

import pytest
import requests

from ollama_service_updated.models import (
    ChatRequest,
    EmbeddingRequest,
    GenerateRequest,
)
from ollama_service_updated.service import (
    OllamaAuthError,
    OllamaConnectionError,
    OllamaService,
)

AVAILABLE_MODELS = [
    "dolphin-llama3",
    "dolphin-phi",
    "llama3.1",
    "deepseek-r1",
    "mistral",
    "gemma2",
    "qwen2.5",
]


@pytest.fixture(scope="module")
def service():
    svc = OllamaService(host="18.205.235.191", port=11434, timeout=120)
    yield svc
    svc.close()


class TestHealthCheck:
    def test_is_healthy(self, service):
        assert service.is_healthy() is True

    def test_base_url(self, service):
        assert service.base_url == "http://18.205.235.191:11434"

    def test_health_endpoint_reachable(self):
        response = requests.get("http://18.205.235.191:11434/", timeout=10)
        assert response.status_code == 200


class TestListModels:
    def test_list_models_returns_list(self, service):
        models = service.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_list_model_names_returns_strings(self, service):
        names = service.list_model_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_expected_models_present(self, service):
        names = service.list_model_names()
        for model in AVAILABLE_MODELS:
            matches = [n for n in names if model in n]
            assert len(matches) > 0, f"Model {model} not found in {names}"

    def test_model_info_fields(self, service):
        models = service.list_models()
        model = models[0]
        assert model.name != ""
        assert model.size > 0
        assert model.digest != ""

    def test_tags_endpoint_direct(self):
        response = requests.get("http://18.205.235.191:11434/api/tags", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0


class TestShowModel:
    def test_show_model(self, service):
        detail = service.show_model("mistral:latest")
        assert detail.details != {}

    def test_show_model_has_template(self, service):
        detail = service.show_model("mistral:latest")
        assert isinstance(detail.template, str)

    def test_show_model_has_parameters(self, service):
        detail = service.show_model("mistral:latest")
        assert isinstance(detail.parameters, str)

    def test_show_endpoint_direct(self):
        response = requests.post(
            "http://18.205.235.191:11434/api/show",
            json={"name": "mistral:latest"},
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()
        assert "details" in data


class TestGenerate:
    def test_generate_simple(self, service):
        response = service.generate_simple("mistral", "Say hello in one word.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_with_request(self, service):
        req = GenerateRequest(
            model="mistral",
            prompt="What is 2+2? Reply with just the number.",
            stream=False,
        )
        resp = service.generate(req)
        assert resp.done is True
        assert resp.model == "mistral"
        assert len(resp.response) > 0

    def test_generate_with_system_prompt(self, service):
        response = service.generate_simple(
            model="mistral",
            prompt="What are you?",
            system="You are a helpful calculator. Reply in one short sentence.",
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_with_options(self, service):
        req = GenerateRequest(
            model="mistral",
            prompt="Say yes.",
            stream=False,
            options={"temperature": 0.1, "num_predict": 10},
        )
        resp = service.generate(req)
        assert resp.done is True
        assert len(resp.response) > 0

    def test_generate_endpoint_direct(self):
        response = requests.post(
            "http://18.205.235.191:11434/api/generate",
            json={"model": "mistral", "prompt": "Say ok.", "stream": False},
            timeout=120,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["done"] is True
        assert len(data["response"]) > 0


class TestChat:
    def test_chat_simple(self, service):
        response = service.chat_simple("mistral", "Say hello in one word.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_with_request(self, service):
        req = ChatRequest(
            model="mistral",
            messages=[{"role": "user", "content": "What is 1+1? Reply with just the number."}],
            stream=False,
        )
        resp = service.chat(req)
        assert resp.done is True
        assert resp.message.role == "assistant"
        assert len(resp.message.content) > 0

    def test_chat_with_system_message(self, service):
        response = service.chat_simple(
            model="mistral",
            user_message="What are you?",
            system_message="You are a pirate. Reply in one short sentence.",
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_multi_turn(self, service):
        messages = [
            {"role": "user", "content": "My name is TestBot."},
            {"role": "assistant", "content": "Hello TestBot!"},
            {"role": "user", "content": "What is my name? Reply with just the name."},
        ]
        req = ChatRequest(model="mistral", messages=messages, stream=False)
        resp = service.chat(req)
        assert resp.done is True
        assert len(resp.message.content) > 0

    def test_chat_endpoint_direct(self):
        response = requests.post(
            "http://18.205.235.191:11434/api/chat",
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": "Say ok."}],
                "stream": False,
            },
            timeout=120,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["done"] is True
        assert "message" in data


class TestEmbeddings:
    def test_get_embedding(self, service):
        embedding = service.get_embedding("mistral", "Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)

    def test_embedding_request(self, service):
        req = EmbeddingRequest(model="mistral", input="Test embedding")
        resp = service.embeddings(req)
        assert resp.model == "mistral"
        assert len(resp.embeddings) > 0
        assert len(resp.embeddings[0]) > 0

    def test_embedding_dimensions_consistent(self, service):
        e1 = service.get_embedding("mistral", "First text")
        e2 = service.get_embedding("mistral", "Second text")
        assert len(e1) == len(e2)

    def test_embed_endpoint_direct(self):
        response = requests.post(
            "http://18.205.235.191:11434/api/embed",
            json={"model": "mistral", "input": "test"},
            timeout=120,
        )
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data


class TestListRunning:
    def test_list_running_returns_list(self, service):
        running = service.list_running()
        assert isinstance(running, list)

    def test_ps_endpoint_direct(self):
        response = requests.get("http://18.205.235.191:11434/api/ps", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data


class TestMultipleModels:
    @pytest.mark.parametrize("model_name", ["dolphin-llama3", "dolphin-phi", "gemma2", "qwen2.5"])
    def test_generate_across_models(self, service, model_name):
        response = service.generate_simple(model_name, "Say hi in one word.")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.parametrize("model_name", ["mistral", "llama3.1"])
    def test_chat_across_models(self, service, model_name):
        response = service.chat_simple(model_name, "Say hi in one word.")
        assert isinstance(response, str)
        assert len(response) > 0


class TestContextManager:
    def test_context_manager(self):
        with OllamaService(host="18.205.235.191", port=11434) as svc:
            assert svc.is_healthy() is True


class TestConnectionError:
    def test_bad_host_is_not_healthy(self):
        svc = OllamaService(host="192.0.2.1", port=99999, timeout=3)
        assert svc.is_healthy() is False
        svc.close()


class TestAuthentication:
    def test_no_auth_by_default(self):
        svc = OllamaService(host="18.205.235.191", port=11434)
        assert svc.authenticated is False
        svc.close()

    def test_api_key_sets_bearer_header(self):
        svc = OllamaService(host="18.205.235.191", port=11434, api_key="test-key-123")
        assert svc.authenticated is True
        assert svc._session.headers["Authorization"] == "Bearer test-key-123"
        svc.close()

    def test_basic_auth_configured(self):
        svc = OllamaService(
            host="18.205.235.191", port=11434, username="admin", password="secret"
        )
        assert svc.authenticated is True
        assert svc._session.auth == ("admin", "secret")
        svc.close()

    def test_custom_headers(self):
        svc = OllamaService(
            host="18.205.235.191",
            port=11434,
            headers={"X-Custom-Auth": "my-token", "X-Request-Source": "test"},
        )
        assert svc._session.headers["X-Custom-Auth"] == "my-token"
        assert svc._session.headers["X-Request-Source"] == "test"
        svc.close()

    def test_api_key_and_custom_headers_combined(self):
        svc = OllamaService(
            host="18.205.235.191",
            port=11434,
            api_key="my-api-key",
            headers={"X-Trace-Id": "abc-123"},
        )
        assert svc._session.headers["Authorization"] == "Bearer my-api-key"
        assert svc._session.headers["X-Trace-Id"] == "abc-123"
        assert svc.authenticated is True
        svc.close()

    def test_service_works_without_auth(self):
        svc = OllamaService(host="18.205.235.191", port=11434)
        assert svc.is_healthy() is True
        models = svc.list_models()
        assert len(models) > 0
        svc.close()

    def test_verify_auth_passes_without_key(self):
        svc = OllamaService(host="18.205.235.191", port=11434)
        assert svc.verify_auth() is True
        svc.close()

    def test_verify_auth_hits_tags_endpoint(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.verify_auth()
        assert "REQUEST  | method=GET | url=http://18.205.235.191:11434/api/tags" in caplog.text
        assert "action=verify_auth" in caplog.text
        svc.close()

    def test_verify_auth_logs_response(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.verify_auth()
        assert "RESPONSE | method=GET | url=http://18.205.235.191:11434/api/tags | status=200" in caplog.text
        svc.close()

    def test_verify_auth_with_bad_host_raises(self):
        svc = OllamaService(host="192.0.2.1", port=99999, timeout=3)
        with pytest.raises(Exception):
            svc.verify_auth()
        svc.close()


class TestLogging:
    def test_logger_exists(self):
        log = logging.getLogger("ollama_service")
        assert log is not None

    def test_init_logs_base_url(self, caplog):
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc = OllamaService(host="18.205.235.191", port=11434)
        assert "Initialized OllamaService at http://18.205.235.191:11434" in caplog.text
        svc.close()

    def test_api_key_auth_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc = OllamaService(host="18.205.235.191", port=11434, api_key="key-abc")
        assert "Bearer token authentication" in caplog.text
        svc.close()

    def test_basic_auth_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc = OllamaService(
                host="18.205.235.191", port=11434, username="user1", password="pass1"
            )
        assert "basic authentication" in caplog.text
        assert "user1" in caplog.text
        svc.close()

    def test_request_logged_with_method_and_url(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.list_running()
        assert "REQUEST  | method=GET | url=http://18.205.235.191:11434/api/ps" in caplog.text
        svc.close()

    def test_response_logged_with_status_and_elapsed(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.list_running()
        assert "RESPONSE | method=GET | url=http://18.205.235.191:11434/api/ps | status=200" in caplog.text
        assert "elapsed=" in caplog.text
        assert "bytes" in caplog.text
        svc.close()

    def test_list_models_result_logged(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.list_models()
        assert "RESULT   | action=list_models | count=" in caplog.text
        assert "REQUEST  | method=GET | url=http://18.205.235.191:11434/api/tags" in caplog.text
        assert "RESPONSE | method=GET | url=http://18.205.235.191:11434/api/tags | status=200" in caplog.text
        svc.close()

    def test_generate_logged_with_full_metrics(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.generate_simple("mistral", "Say ok.")
        assert "REQUEST  | method=POST | url=http://18.205.235.191:11434/api/generate" in caplog.text
        assert "RESPONSE | method=POST | url=http://18.205.235.191:11434/api/generate | status=200" in caplog.text
        assert "RESULT   | action=generate | model=mistral" in caplog.text
        assert "eval_count=" in caplog.text
        assert "total_duration_ns=" in caplog.text
        assert "response_length=" in caplog.text
        svc.close()

    def test_chat_logged_with_full_metrics(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.chat_simple("mistral", "Say ok.")
        assert "REQUEST  | method=POST | url=http://18.205.235.191:11434/api/chat" in caplog.text
        assert "RESPONSE | method=POST | url=http://18.205.235.191:11434/api/chat | status=200" in caplog.text
        assert "RESULT   | action=chat | model=mistral" in caplog.text
        assert "messages=" in caplog.text
        assert "eval_count=" in caplog.text
        assert "response_length=" in caplog.text
        svc.close()

    def test_embeddings_logged(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.get_embedding("mistral", "test")
        assert "REQUEST  | method=POST | url=http://18.205.235.191:11434/api/embed" in caplog.text
        assert "RESPONSE | method=POST | url=http://18.205.235.191:11434/api/embed | status=200" in caplog.text
        assert "RESULT   | action=embeddings | model=mistral" in caplog.text
        assert "dimensions=" in caplog.text
        svc.close()

    def test_health_check_logged_with_url(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.is_healthy()
        assert "REQUEST  | method=GET | url=http://18.205.235.191:11434/" in caplog.text
        assert "RESPONSE | method=GET | url=http://18.205.235.191:11434/" in caplog.text
        assert "healthy=True" in caplog.text
        svc.close()

    def test_close_logged(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.close()
        assert "Closing OllamaService session" in caplog.text

    def test_auth_status_in_request_log(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434, api_key="test-key")
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.is_healthy()
        assert "authenticated=True" in caplog.text
        svc.close()

    def test_no_auth_status_in_request_log(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.is_healthy()
        assert "authenticated=False" in caplog.text
        svc.close()

    def test_payload_keys_logged(self, caplog):
        svc = OllamaService(host="18.205.235.191", port=11434)
        with caplog.at_level(logging.INFO, logger="ollama_service"):
            svc.show_model("mistral:latest")
        assert "payload_keys=['name']" in caplog.text
        svc.close()

    def test_failed_connection_logged_with_elapsed(self, caplog):
        svc = OllamaService(host="192.0.2.1", port=99999, timeout=3)
        with caplog.at_level(logging.WARNING, logger="ollama_service"):
            svc.is_healthy()
        assert "FAILURE  | method=GET | url=http://192.0.2.1:99999/" in caplog.text
        assert "elapsed=" in caplog.text
        svc.close()


NGINX_HOST = "18.205.235.191"
NGINX_PORT = 8080
VALID_API_KEY = "ollama-key-001"
INVALID_API_KEY = "wrong-key-999"
BASIC_AUTH_USER = "admin"
BASIC_AUTH_PASS = "ollama-admin-pass"


@pytest.fixture(scope="module")
def nginx_service():
    svc = OllamaService(host=NGINX_HOST, port=NGINX_PORT, api_key=VALID_API_KEY, timeout=120)
    yield svc
    svc.close()


class TestNginxBearerAuth:
    def test_valid_bearer_key_health(self):
        response = requests.get(
            "http://18.205.235.191:8080/health",
            timeout=10,
        )
        assert response.status_code == 200

    def test_valid_bearer_key_list_models(self):
        response = requests.get(
            "http://18.205.235.191:8080/api/tags",
            headers={"Authorization": "Bearer ollama-key-001"},
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    def test_valid_bearer_key_generate(self):
        response = requests.post(
            "http://18.205.235.191:8080/api/generate",
            headers={"Authorization": "Bearer ollama-key-001"},
            json={"model": "mistral", "prompt": "Say ok.", "stream": False},
            timeout=120,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["done"] is True

    def test_valid_bearer_key_chat(self):
        response = requests.post(
            "http://18.205.235.191:8080/api/chat",
            headers={"Authorization": "Bearer ollama-key-002"},
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": "Say ok."}],
                "stream": False,
            },
            timeout=120,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["done"] is True

    def test_valid_bearer_key_show(self):
        response = requests.post(
            "http://18.205.235.191:8080/api/show",
            headers={"Authorization": "Bearer ollama-key-003"},
            json={"name": "mistral:latest"},
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()
        assert "details" in data

    def test_valid_bearer_key_embeddings(self):
        response = requests.post(
            "http://18.205.235.191:8080/api/embed",
            headers={"Authorization": "Bearer ollama-key-001"},
            json={"model": "mistral", "input": "test"},
            timeout=120,
        )
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data

    def test_invalid_bearer_key_rejected(self):
        response = requests.get(
            "http://18.205.235.191:8080/api/tags",
            headers={"Authorization": "Bearer wrong-key-999"},
            timeout=10,
        )
        assert response.status_code == 401

    def test_no_auth_header_rejected(self):
        response = requests.get(
            "http://18.205.235.191:8080/api/tags",
            timeout=10,
        )
        assert response.status_code == 401

    def test_empty_bearer_rejected(self):
        response = requests.get(
            "http://18.205.235.191:8080/api/tags",
            headers={"Authorization": "Bearer "},
            timeout=10,
        )
        assert response.status_code == 401


class TestNginxBasicAuth:
    def test_valid_basic_auth_list_models(self):
        response = requests.get(
            "http://18.205.235.191:8080/api/tags",
            auth=("admin", "ollama-admin-pass"),
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    def test_valid_basic_auth_generate(self):
        response = requests.post(
            "http://18.205.235.191:8080/api/generate",
            auth=("admin", "ollama-admin-pass"),
            json={"model": "mistral", "prompt": "Say ok.", "stream": False},
            timeout=120,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["done"] is True

    def test_valid_basic_auth_second_user(self):
        response = requests.get(
            "http://18.205.235.191:8080/api/tags",
            auth=("user", "ollama-user-pass"),
            timeout=10,
        )
        assert response.status_code == 200

    def test_wrong_password_rejected(self):
        response = requests.get(
            "http://18.205.235.191:8080/api/tags",
            auth=("admin", "wrong-password"),
            timeout=10,
        )
        assert response.status_code == 401

    def test_unknown_user_rejected(self):
        response = requests.get(
            "http://18.205.235.191:8080/api/tags",
            auth=("nobody", "anything"),
            timeout=10,
        )
        assert response.status_code == 401


class TestNginxHealthEndpoint:
    def test_health_no_auth_required(self):
        response = requests.get(
            "http://18.205.235.191:8080/health",
            timeout=10,
        )
        assert response.status_code == 200

    def test_health_with_auth_still_works(self):
        response = requests.get(
            "http://18.205.235.191:8080/health",
            headers={"Authorization": "Bearer ollama-key-001"},
            timeout=10,
        )
        assert response.status_code == 200


class TestNginxServiceIntegration:
    def test_service_with_valid_bearer_key(self):
        svc = OllamaService(host="18.205.235.191", port=8080, api_key="ollama-key-001")
        assert svc.verify_auth() is True
        models = svc.list_models()
        assert len(models) > 0
        svc.close()

    def test_service_with_valid_basic_auth(self):
        svc = OllamaService(
            host="18.205.235.191", port=8080, username="admin", password="ollama-admin-pass"
        )
        assert svc.verify_auth() is True
        names = svc.list_model_names()
        assert len(names) > 0
        svc.close()

    def test_service_with_invalid_key_fails_verify(self):
        svc = OllamaService(host="18.205.235.191", port=8080, api_key="wrong-key-999")
        assert svc.verify_auth() is False
        svc.close()

    def test_service_no_auth_fails_verify(self):
        svc = OllamaService(host="18.205.235.191", port=8080)
        assert svc.verify_auth() is False
        svc.close()

    def test_service_generate_through_nginx(self):
        svc = OllamaService(host="18.205.235.191", port=8080, api_key="ollama-key-001")
        response = svc.generate_simple("mistral", "Say ok in one word.")
        assert isinstance(response, str)
        assert len(response) > 0
        svc.close()

    def test_service_chat_through_nginx(self):
        svc = OllamaService(host="18.205.235.191", port=8080, api_key="ollama-key-002")
        response = svc.chat_simple("mistral", "Say ok in one word.")
        assert isinstance(response, str)
        assert len(response) > 0
        svc.close()

    def test_service_embeddings_through_nginx(self):
        svc = OllamaService(host="18.205.235.191", port=8080, api_key="ollama-key-001")
        embedding = svc.get_embedding("mistral", "test")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        svc.close()

    def test_service_show_model_through_nginx(self):
        svc = OllamaService(
            host="18.205.235.191", port=8080, username="admin", password="ollama-admin-pass"
        )
        detail = svc.show_model("mistral:latest")
        assert detail.details != {}
        svc.close()

    def test_invalid_key_raises_on_request(self):
        svc = OllamaService(host="18.205.235.191", port=8080, api_key="bad-key")
        with pytest.raises(OllamaAuthError):
            svc.list_models()
        svc.close()

    def test_no_auth_raises_on_request(self):
        svc = OllamaService(host="18.205.235.191", port=8080)
        with pytest.raises(OllamaAuthError):
            svc.list_models()
        svc.close()
