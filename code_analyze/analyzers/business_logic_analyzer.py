import ast
from typing import List, Dict, Any
import networkx as nx
from .file_analyzer import FileAnalyzer

class BusinessLogicAnalyzer:
    def __init__(self, llm_api):
        self.llm_api = llm_api
        self.dependency_graph = nx.DiGraph()
        self.file_analyzer = FileAnalyzer(llm_api)

    async def analyze_business_logic(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze business logic across multiple files in the codebase.
        """
        business_components = {}
        
        # First pass: Extract individual file business logic
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST and analyze file
            tree = ast.parse(content)
            file_components = await self.file_analyzer.analyze_file(tree, file_path)
            business_components[file_path] = file_components
            
            # Build dependency graph
            self._build_dependency_graph(tree, file_path)
            
        # Second pass: Analyze cross-file relationships
        business_flows = self._analyze_business_flows(business_components)
        
        # Generate high-level summary
        summary_prompt = self._create_business_logic_summary_prompt(business_components, business_flows)
        overall_summary = await self.llm_api.analyze_code(summary_prompt)
        
        return {
            "components": business_components,
            "flows": business_flows,
            "summary": overall_summary,
            "dependency_graph": self._serialize_graph()
        }

    def _build_dependency_graph(self, tree: ast.AST, file_path: str):
        """Build a dependency graph between files based on imports."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = node.names[0].name if isinstance(node, ast.Import) else node.module
                self.dependency_graph.add_edge(file_path, module_name)

    def _analyze_business_flows(self, components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze business flows across different files."""
        flows = []
        visited = set()
        
        for start_node in self.dependency_graph.nodes():
            if start_node not in visited:
                flow = self._trace_business_flow(start_node, components, visited)
                if flow:
                    flows.append(flow)
                    
        return flows

    def _trace_business_flow(self, start_node: str, components: Dict[str, Any], visited: set) -> Dict[str, Any]:
        """Trace a business flow starting from a specific node."""
        if start_node in visited or start_node not in components:
            return None
            
        visited.add(start_node)
        flow = {
            "start": start_node,
            "steps": [],
            "related_components": []
        }
        
        file_components = components[start_node]
        flow["related_components"].extend(file_components["functions"])
        flow["related_components"].extend(file_components["classes"])
        
        for next_node in self.dependency_graph.successors(start_node):
            sub_flow = self._trace_business_flow(next_node, components, visited)
            if sub_flow:
                flow["steps"].append(sub_flow)
                
        return flow

    def _create_business_logic_summary_prompt(self, components: Dict[str, Any], flows: List[Dict[str, Any]]) -> str:
        """Create a prompt for LLM to generate overall business logic summary."""
        prompt = "Analyze the following business components and their relationships:\n\n"
        
        for file_path, file_components in components.items():
            prompt += f"\nFile: {file_path}\n"
            prompt += "Functions:\n"
            for func in file_components["functions"]:
                prompt += f"- {func['name']}: {func['logic']}\n"
            prompt += "Classes:\n"
            for cls in file_components["classes"]:
                prompt += f"- {cls['name']}: {cls['logic']}\n"
                
        prompt += "\nBusiness Flows:\n"
        for flow in flows:
            prompt += f"Flow starting from {flow['start']}:\n"
            prompt += f"- Related components: {', '.join(c['name'] for c in flow['related_components'])}\n"
            
        prompt += "\nProvide a comprehensive summary of the business logic, including:\n"
        prompt += "1. Main business components and their responsibilities\n"
        prompt += "2. Key business flows and their interactions\n"
        prompt += "3. Important business rules and constraints\n"
        
        return prompt

    def _serialize_graph(self) -> Dict[str, Any]:
        """Serialize the dependency graph for output."""
        return {
            "nodes": list(self.dependency_graph.nodes()),
            "edges": list(self.dependency_graph.edges())
        }
