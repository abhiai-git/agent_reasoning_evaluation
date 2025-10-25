from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    provider: str                  # 'openai' | 'gemini' | 'claude' | 'bedrock'
    model: str
    api_key: Optional[str] = None
    aws_profile: Optional[str] = None
