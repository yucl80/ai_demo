from typing import Dict, Any
import json
from .base_expert import ExpertModel

class SecurityExpert(ExpertModel):
    """Expert model for analyzing code security and vulnerabilities."""
    
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a security expert, analyze the given code and context. Focus on:
        1. Potential security vulnerabilities
        2. Adherence to security best practices
        3. Data handling and privacy concerns
        4. Recommendations for security improvements

        Code:
        {code}

        Context:
        {context}

        Provide a detailed security analysis of the code.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))
