from pydantic import BaseModel
from typing import List, Optional

class jobData(BaseModel):
    role: str
    skills: List[str]
    is_valid_document: bool 
    summary: Optional[str] = None
    
    