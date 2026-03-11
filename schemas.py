from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class GeminiContentPart(BaseModel):
    text: Optional[str] = None
    inline_data: Optional[Any] = None
    function_call: Optional[Any] = None
    function_response: Optional[Any] = None

class GeminiContent(BaseModel):
    role: str  # "user" или "model"
    parts: List[GeminiContentPart]

class ProxyRequest(BaseModel):
    # Вместо одного prompt используем contents для поддержки истории
    contents: List[GeminiContent] 
    system_instruction: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None  # Для Function Calling
    
    # Параметры генерации
    temperature: Optional[float] = 0.7
    max_output_tokens: Optional[int] = 1000
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    
    # Безопасность
    safety_settings: Optional[List[Dict[str, str]]] = None

class ProxyResponse(BaseModel):
    answer: str
    function_calls: Optional[List[Dict[str, Any]]] = None
    model_used: str
    finish_reason: str
    
class EmbeddingRequest(BaseModel):
    text: str
    task_type: Optional[str] = "RETRIEVAL_QUERY" # Или "RETRIEVAL_DOCUMENT" для сохранения в БД
    title: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model_used: str