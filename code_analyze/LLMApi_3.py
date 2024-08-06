import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import json
import ast
import networkx as nx
from typing import List, Dict, Any

api_key = 'your_openai_api_key'

# 初始化OpenAI的API客户端
base_url = "http://127.0.0.1:8000/v1/"

from openai import OpenAI
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url=base_url, api_key=api_key)

class LLMApi:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as an environment variable or pass it to the constructor.")
        openai.api_key = self.api_key
        self.model = model
        self.global_context = {}
        self.analysis_graph = nx.DiGraph()
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        with open('D:\\workspaces\\python_projects\\code_review\\prompts.json', 'r') as f:
            return json.load(f)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout= 180000,
            )
            print(response)
            return response.choices[0].message.content.strip()
        except RuntimeError as e:
            print(f"OpenAI API error: {e}")
            raise

    async def analyze(self, content: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert code analyst and software architect."},
            {"role": "user", "content": f"{prompt}\n\nCode:\n{content}"}
        ]
        return await self._call_openai_api(messages)

    async def analyze_with_global_context(self, code: str) -> Dict[str, Any]:
        self.global_context['overview'] = await self.analyze(code, self.prompts['global_overview'])

        segments = self._segment_code(code)
        self._build_dependency_graph(segments)
        detailed_analyses = await self._analyze_segments_with_context(segments)
        summary = await self._summarize_analysis(detailed_analyses)

        return {
            'overview': self.global_context['overview'],
            'detailed_analyses': detailed_analyses,
            'summary': summary,
            'dependency_graph': nx.node_link_data(self.analysis_graph)
        }

    def _segment_code(self, code: str) -> List[Dict[str, str]]:
        tree = ast.parse(code)
        segments = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                segment = {
                    'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                    'name': node.name,
                    'code': ast.get_source_segment(code, node),
                    'docstring': ast.get_docstring(node)
                }
                segments.append(segment)

        if not segments:
            segments.append({
                'type': 'module',
                'name': 'main',
                'code': code,
                'docstring': ast.get_docstring(tree)
            })

        return segments

    def _build_dependency_graph(self, segments: List[Dict[str, str]]):
        for segment in segments:
            self.analysis_graph.add_node(segment['name'], type=segment['type'])

        for segment in segments:
            tree = ast.parse(segment['code'])
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    for other_segment in segments:
                        if other_segment['name'] == node.id:
                            self.analysis_graph.add_edge(segment['name'], other_segment['name'])

    async def _analyze_segments_with_context(self, segments: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        analyses = []
        for segment in segments:
            context = self._get_segment_context(segment)
            analysis_prompt = self.prompts['segment_analysis'].format(
                segment_type=segment['type'],
                segment_name=segment['name'],
                global_context=json.dumps(self.global_context, indent=2),
                segment_context=json.dumps(context, indent=2),
                segment_code=segment['code']
            )
            analysis = await self.analyze(segment['code'], analysis_prompt)
            analyses.append({
                'type': segment['type'],
                'name': segment['name'],
                'analysis': analysis
            })
            self.global_context[segment['name']] = analysis[:200]  # Store a summary

        return analyses

    def _get_segment_context(self, segment: Dict[str, str]) -> Dict[str, Any]:
        predecessors = list(self.analysis_graph.predecessors(segment['name']))
        successors = list(self.analysis_graph.successors(segment['name']))
        return {
            'type': segment['type'],
            'name': segment['name'],
            'docstring': segment['docstring'],
            'dependencies': predecessors,
            'dependents': successors
        }

    async def _summarize_analysis(self, detailed_analyses: List[Dict[str, Any]]) -> str:
        summary_prompt = self.prompts['summary_analysis'].format(
            global_context=json.dumps(self.global_context, indent=2),
            detailed_analyses=json.dumps(detailed_analyses, indent=2)
        )
        return await self.analyze("", summary_prompt)

    async def interactive_analysis(self, code: str):
        print("Analyzing code and building context...")
        analysis_result = await self.analyze_with_global_context(code)
        print("Analysis complete. You can now ask questions about the code.")
        
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break

            prompt = self.prompts['interactive_query'].format(
                overview=analysis_result['overview'],
                detailed_analyses=json.dumps(analysis_result['detailed_analyses'], indent=2),
                summary=analysis_result['summary'],
                query=query
            )
            
            answer = await self.analyze("", prompt)
            print("\nAnswer:", answer)

# Example usage
async def main():
    api_key = "your-api-key-here"  # Replace with your actual API key
    llm_api = LLMApi(api_key)
    
    code_to_analyze = """
import math
def calculate_circle_area(radius):
    #Calculate the area of a circle given its radius.
    return math.pi * radius ** 2

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def calculate_area(self):
        #Calculate the area of the rectangle.
        return self.width * self.height

def main():
    # Calculate circle area
    circle_radius = 5
    circle_area = calculate_circle_area(circle_radius)
    print(f"Area of circle with radius {circle_radius}: {circle_area:.2f}")

    # Calculate rectangle area
    rect = Rectangle(4, 6)
    rect_area = rect.calculate_area()
    print(f"Area of rectangle with width {rect.width} and height {rect.height}: {rect_area}")

if __name__ == "__main__":
    main()
"""

    # Perform analysis with global context
    analysis_result = await llm_api.analyze_with_global_context(code_to_analyze)
    print(json.dumps(analysis_result, indent=2))

    # Start interactive analysis session
    await llm_api.interactive_analysis(code_to_analyze)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())