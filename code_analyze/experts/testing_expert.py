from typing import Dict, Any
import json
from .base_expert import ExpertModel

class TestingExpert(ExpertModel):
    """Expert model for analyzing code testing aspects."""
    
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a testing expert, analyze the given code and context. Focus on:
        1. Test coverage and quality
        2. Potential edge cases and error scenarios
        3. Testability of the code
        4. Suggestions for improving test suite

        Code:
        {code}

        Context:
        {context}

        Provide a detailed analysis of the code's testing aspects.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))
