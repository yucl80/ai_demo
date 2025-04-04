from abc import ABC, abstractmethod
from typing import Any, Dict

class ExpertModel(ABC):
    """Base class for all expert models."""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        """Analyze code with given context and return analysis results."""
        pass
