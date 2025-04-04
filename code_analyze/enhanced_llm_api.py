import asyncio
import json
from typing import Dict, Any, Tuple
from experts import (
    ExpertModel,
    ArchitectureExpert,
    PerformanceExpert,
    SecurityExpert,
    CodeQualityExpert,
    TestingExpert,
    CodeSummaryExpert
)

class EnhancedLLMApi:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.experts = {
            "architecture": ArchitectureExpert(api_key),
            "performance": PerformanceExpert(api_key),
            "security": SecurityExpert(api_key),
            "code_quality": CodeQualityExpert(api_key),
            "testing": TestingExpert(api_key),
            "code_summary": CodeSummaryExpert(api_key)
        }

    async def enhanced_analysis(self, code: str, repo_path: str = None) -> Dict[str, Any]:
        """Perform enhanced analysis using multiple expert models."""
        # Get basic analysis results
        basic_analysis = await self.analyze_with_global_context(code)
        
        # Prepare enhanced context with various analyses
        enhanced_context = {
            "ast_analysis": self.ast_analyzer.analyze(code),
            "git_analysis": self.git_analyzer.analyze(repo_path) if repo_path else None,
            "static_analysis": self.static_analyzer.analyze(code),
            "test_analysis": self.test_analyzer.analyze(repo_path) if repo_path else None,
            "dependency_analysis": self.dependency_analyzer.analyze(repo_path) if repo_path else None,
            "pattern_analysis": self.pattern_recognizer.recognize(code),
            "performance_analysis": self.performance_analyzer.analyze(code),
            "security_analysis": self.security_scanner.scan(code),
            "domain_insights": self.domain_knowledge.get_insights(code),
            "comments_analysis": self.comment_extractor.extract(code),
            "metaprogramming_analysis": self.metaprogramming_analyzer.analyze(code),
            "environment_analysis": self.environment_analyzer.analyze(repo_path) if repo_path else None
        }

        # Run expert analyses
        expert_analyses = await self._run_expert_analyses(code, enhanced_context)
        enhanced_context.update(expert_analyses)
        
        # Generate code summary
        code_summary = expert_analyses['code_summary_analysis']

        # Integrate all analyses
        final_analysis = await self._integrate_analyses(enhanced_context)

        # Generate visualization
        visualization = self.generate_visualization(enhanced_context)

        return {
            **basic_analysis, 
            "enhanced_analysis": final_analysis,
            "expert_analyses": expert_analyses,
            "code_summary": code_summary,
            "visualization": visualization
        }

    async def _run_expert_analyses(self, code: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Run analyses using all expert models concurrently."""
        expert_analyses = {}
        tasks = []
        for expert_name, expert in self.experts.items():
            tasks.append(self._run_expert_analysis(expert_name, expert, code, context))
        results = await asyncio.gather(*tasks)
        for expert_name, analysis in results:
            expert_analyses[f"{expert_name}_analysis"] = analysis
        return expert_analyses

    async def _run_expert_analysis(self, expert_name: str, expert: ExpertModel, code: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Run analysis using a single expert model."""
        analysis = await expert.analyze(code, context)
        return expert_name, analysis

    async def _integrate_analyses(self, context: Dict[str, Any]) -> str:
        """Integrate all analyses into a comprehensive report."""
        integration_prompt = """
        As a senior software architect and code analyst, review and integrate the following analyses of a codebase:

        {context}

        Provide a comprehensive, high-level summary of the codebase, addressing:
        1. Overall architecture and design
        2. Code quality and maintainability
        3. Performance characteristics
        4. Security considerations
        5. Testing and reliability
        6. Areas for improvement and recommendations
        
        Also, consider the provided code summary and how it relates to the detailed analyses.

        Your summary should synthesize insights from all the expert analyses and provide a holistic view of the codebase.
        """
        return await self.analyze("", integration_prompt.format(context=json.dumps(context)))
