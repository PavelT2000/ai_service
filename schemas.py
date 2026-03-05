from pydantic import BaseModel, Field
from typing import Optional

class UserRequest(BaseModel):
    prompt: str
    system_instruction: Optional[str] = "Ты полезный ассистент."
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = 1000
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    # Параметры цензуры (можно передавать строками)
    hate_threshold: Optional[str] = "BLOCK_MEDIUM_AND_ABOVE"
    harassment_threshold: Optional[str] = "BLOCK_MEDIUM_AND_ABOVE"
    
class AIResponse(BaseModel):
    answer: Optional[str] = ""  # Теперь он может быть None или пустой строкой
    model: str