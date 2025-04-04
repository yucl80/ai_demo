from typing import Dict, Any
import json
from .base_expert import ExpertModel

class CodeSummaryExpert(ExpertModel):
    """Expert model for generating code summaries."""
    
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a code summarization expert, analyze the given code and context. Your task is to generate a high-level summary of the code that captures its main functionality, structure, and purpose. Focus on:

        1. The overall purpose of the code
        2. Key components or modules and their roles
        3. Main algorithms or processes implemented
        4. Important data structures used
        5. External dependencies and their purposes
        6. Any notable design patterns or architectural choices

        Provide a concise yet informative summary that would help a developer quickly understand the essence of this code without delving into every detail.

        Code:
        {code}

        Context:
        {context}

        Generate a comprehensive summary of the code in about 200-300 words.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))
