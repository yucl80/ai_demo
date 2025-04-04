import ast
from typing import Dict, Any

class FileAnalyzer:
    def __init__(self, llm_api):
        self.llm_api = llm_api

    async def analyze_file(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze a single file's business logic components."""
        components = {
            "functions": [],
            "classes": [],
            "business_rules": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                await self._analyze_function(node, components, file_path)
            elif isinstance(node, ast.ClassDef):
                await self._analyze_class(node, components, file_path)
                
        return components

    async def _analyze_function(self, node: ast.FunctionDef, components: Dict[str, Any], file_path: str):
        """Analyze a function's business logic."""
        function_code = ast.get_source_segment(self._get_file_content(file_path), node)
        function_prompt = f"Analyze the business logic in this function:\n{function_code}"
        function_analysis = await self.llm_api.analyze_code(function_prompt)
        
        components["functions"].append({
            "name": node.name,
            "logic": function_analysis,
            "code": function_code
        })

    async def _analyze_class(self, node: ast.ClassDef, components: Dict[str, Any], file_path: str):
        """Analyze a class's business logic."""
        class_code = ast.get_source_segment(self._get_file_content(file_path), node)
        class_prompt = f"Analyze the business logic in this class:\n{class_code}"
        class_analysis = await self.llm_api.analyze_code(class_prompt)
        
        components["classes"].append({
            "name": node.name,
            "logic": class_analysis,
            "code": class_code
        })

    def _get_file_content(self, file_path: str) -> str:
        """Helper method to get file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
