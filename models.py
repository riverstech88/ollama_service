from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class ModelInfo:
    name: str
    model: str
    size: int
    digest: str
    modified_at: str
    details: dict = field(default_factory=dict)


@dataclass
class ModelDetail:
    modelfile: str
    parameters: str
    template: str
    details: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)


@dataclass
class GenerateRequest:
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    stream: bool = False
    options: Optional[dict] = None


@dataclass
class GenerateResponse:
    model: str
    response: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    context: Optional[list] = None


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatRequest:
    model: str
    messages: list
    stream: bool = False
    options: Optional[dict] = None


@dataclass
class ChatResponse:
    model: str
    message: ChatMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


@dataclass
class EmbeddingRequest:
    model: str
    input: Union[str, list[str]]


@dataclass
class EmbeddingResponse:
    model: str
    embeddings: list[list[float]]


@dataclass
class PullRequest:
    name: str
    stream: bool = False


@dataclass
class PullResponse:
    status: str
