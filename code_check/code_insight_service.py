import ast
import json
import networkx as nx
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os

class CodeInsightService:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        openai.api_key = self.api_key
        self.model = model
        self.dependency_graph = nx.DiGraph()
        self.code_structure = {}
        
    def parse_code(self, file_path: str) -> Dict[str, Any]:
        """Parse Python code and extract its structure"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
            structure = {
                'imports': [],
                'classes': [],
                'functions': [],
                'global_vars': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    structure['imports'].extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    structure['imports'].append(f"{node.module}.{node.names[0].name}")
                elif isinstance(node, ast.ClassDef):
                    structure['classes'].append({
                        'name': node.name,
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        'line_no': node.lineno
                    })
                elif isinstance(node, ast.FunctionDef):
                    if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                        structure['functions'].append({
                            'name': node.name,
                            'line_no': node.lineno,
                            'args': [arg.arg for arg in node.args.args]
                        })
                        
            return structure
        except SyntaxError as e:
            return {'error': f"Syntax error in file: {str(e)}"}
            
    def analyze_dependencies(self, file_path: str) -> Dict[str, List[str]]:
        """Analyze code dependencies"""
        structure = self.parse_code(file_path)
        dependencies = {
            'direct_imports': structure.get('imports', []),
            'internal_deps': [],
            'external_deps': []
        }
        
        for imp in structure.get('imports', []):
            if imp.startswith('.') or not '.' in imp:
                dependencies['internal_deps'].append(imp)
            else:
                dependencies['external_deps'].append(imp.split('.')[0])
                
        return dependencies
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def get_code_insights(self, code: str, context: str = "") -> Dict[str, Any]:
        """Get AI-powered insights about the code"""
        messages = [
            {
                "role": "system",
                "content": """You are an expert code analyst. Analyze the provided code and provide insights about:
                1. Code quality and best practices
                2. Potential improvements
                3. Security considerations
                4. Performance optimization opportunities
                5. Architecture suggestions
                Format your response as a JSON object with these categories."""
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nCode to analyze:\n{code}"
            }
        ]
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
            
    async def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive analysis of a single file"""
        if not os.path.exists(file_path):
            return {"error": "File not found"}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        structure = self.parse_code(file_path)
        dependencies = self.analyze_dependencies(file_path)
        insights = await self.get_code_insights(content)
        
        return {
            "file_path": file_path,
            "structure": structure,
            "dependencies": dependencies,
            "insights": insights
        }
        
    async def analyze_directory(self, dir_path: str, exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """Analyze all Python files in a directory"""
        if exclude_patterns is None:
            exclude_patterns = ['venv', '__pycache__', '.git']
            
        results = {}
        for root, dirs, files in os.walk(dir_path):
            if any(pattern in root for pattern in exclude_patterns):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    results[file_path] = await self.analyze_file(file_path)
                    
        return results

async def main():
    # Example usage
    service = CodeInsightService()
    
    # Analyze a single file
    file_analysis = await service.analyze_file("path/to/your/file.py")
    print(json.dumps(file_analysis, indent=2))
    
    # Analyze a directory
    dir_analysis = await service.analyze_directory("path/to/your/directory")
    print(json.dumps(dir_analysis, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
