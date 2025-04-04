from typing import Dict, Any
from .base_expert import ExpertModel

class ArchitectureExpert(ExpertModel):
    """Expert model for analyzing code architecture and design patterns."""
    
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As an architecture expert, analyze the given code and context. Focus on:
        1. Overall code structure and design patterns
        2. Modularity and component interactions
        3. Scalability and maintainability of the architecture
        4. Suggestions for architectural improvements

        Code:
        {code}

        Context:
        {context}

        Provide a detailed analysis of the code's architecture.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))
