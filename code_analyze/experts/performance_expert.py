from typing import Dict, Any
import json
from .base_expert import ExpertModel

class PerformanceExpert(ExpertModel):
    """Expert model for analyzing code performance and efficiency."""
    
    async def analyze(self, code: str, context: Dict[str, Any]) -> str:
        prompt = """
        As a performance expert, analyze the given code and context. Focus on:
        1. Algorithmic efficiency
        2. Resource usage (CPU, memory, I/O)
        3. Potential bottlenecks and performance hotspots
        4. Optimization suggestions

        Code:
        {code}

        Context:
        {context}

        Provide a detailed analysis of the code's performance characteristics.
        """
        return await self._call_openai_api(prompt.format(code=code, context=json.dumps(context)))
