from typing import Dict, Any
import json
from .base_expert import ExpertModel

class CodeQualityExpert(ExpertModel):
    """Expert model for analyzing code quality and maintainability."""
    
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a code quality expert, analyze the given code and context. Focus on:
        1. Adherence to coding standards and best practices
        2. Code readability and maintainability
        3. Proper use of comments and documentation
        4. Suggestions for improving code quality

        Code:
        {code}

        Context:
        {context}

        Provide a detailed analysis of the code's quality.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))
