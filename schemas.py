"""Схемы данных Pydantic для API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class GeminiContentPart(BaseModel):
    """Часть содержимого сообщения."""
    text: Optional[str] = None
    inline_data: Optional[Any] = None
    function_call: Optional[Any] = None
    function_response: Optional[Any] = None


class GeminiContent(BaseModel):
    """Сообщение с указанием роли (user/model)."""
    role: str
    parts: List[GeminiContentPart]


class ProxyRequest(BaseModel):
    """Запрос на генерацию контента."""
    contents: List[GeminiContent]
    system_instruction: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = 0.7
    max_output_tokens: Optional[int] = 1000
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    safety_settings: Optional[List[Dict[str, str]]] = None


class ProxyResponse(BaseModel):
    """Ответ генерации контента."""
    answer: str
    function_calls: Optional[List[Dict[str, Any]]] = None
    model_used: str
    finish_reason: str


class EmbeddingRequest(BaseModel):
    """Запрос на получение вектора (embedding)."""
    text: str
    task_type: Optional[str] = "RETRIEVAL_QUERY"
    title: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Ответ с вектором."""
    embedding: List[float]
    model_used: str