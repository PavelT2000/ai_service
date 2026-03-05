from pydantic import BaseModel
from typing import Optional

class UserRequest(BaseModel):
    prompt: str
    
class AIResponse(BaseModel):
    answer: Optional[str] = ""  # Теперь он может быть None или пустой строкой
    model: str