import ast
import networkx as nx
from typing import List, Dict, Any
from collections import deque
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ContextAnchor:
    def __init__(self, name: str, type: str, importance: float, summary: str):
        self.name = name
        self.type = type
        self.importance = importance
        self.summary = summary

class CodeIndexer:
    def __init__(self):
        self.index = {}
        self.tfidf_vectorizer = TfidfVectorizer()

    def add_to_index(self, name: str, content: str, type: str):
        self.index[name] = {
            "content": content,
            "type": type,
            "keywords": self.extract_keywords(content)
        }

    def extract_keywords(self, content: str, top_n: int = 5) -> List[str]:
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        sorted_items = sorted(zip(tfidf_matrix.tocsc().data, feature_names), key=lambda x: x[0], reverse=True)
        return [item[1] for item in sorted_items[:top_n]]

    def search(self, query: str) -> List[Dict[str, Any]]:
        query_vector = self.tfidf_vectorizer.transform([query])
        results = []
        for name, data in self.index.items():
            content_vector = self.tfidf_vectorizer.transform([data['content']])
            similarity = np.dot(query_vector.toarray(), content_vector.toarray().T)[0][0]
            results.append({"name": name, "type": data['type'], "similarity": similarity})
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:5]

class LargeCodeAnalyzer:
    def __init__(self, llm_api):
        self.llm_api = llm_api
        self.max_tokens = 7500
        self.dependency_graph = nx.DiGraph()
        self.context_snapshots = deque(maxlen=10)
        self.global_context = {}
        self.local_contexts = {}
        self.context_size_threshold = 5000
        self.block_summaries = {}
        self.indexer = CodeIndexer()
        self.project_summary = ""
        self.context_anchors = []
        self.critical_paths = []

    async def analyze_large_file(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as file:
            content = file.read()
        
        summary = await self.generate_summary(content)
        self.build_dependency_graph(content)
        code_blocks = self.split_code(content)
        
        entry_point = self.identify_entry_point(content)
        entry_point_analysis = await self.analyze_entry_point(entry_point)
        
        self.identify_critical_paths(entry_point)
        
        block_analyses = []
        for i, block in enumerate(code_blocks):
            block_analysis = await self.analyze_block_with_context(block, summary, i)
            block_analyses.append(block_analysis)
            self.save_context_snapshot(i, block_analysis)
            await self.update_global_context(block_analysis)
            self.update_local_context(block['name'], block_analysis)
            self.block_summaries[block['name']] = block_analysis['analysis'][:200]
            self.indexer.add_to_index(block['name'], block['code'], block['type'])
            
            if self.is_context_anchor(block_analysis):
                anchor = self.create_context_anchor(block_analysis)
                self.context_anchors.append(anchor)
            
            if self.should_restructure_context():
                await self.restructure_context()
        
        self.project_summary = await self.generate_project_summary(summary, block_analyses)
        final_analysis = await self.integrate_analyses(summary, block_analyses, entry_point_analysis)
        
        return {
            "summary": summary,
            "project_summary": self.project_summary,
            "entry_point_analysis": entry_point_analysis,
            "block_analyses": block_analyses,
            "final_analysis": final_analysis,
            "dependency_graph": nx.node_link_data(self.dependency_graph),
            "global_context": self.global_context,
            "index": self.indexer.index,
            "context_anchors": [vars(anchor) for anchor in self.context_anchors],
            "critical_paths": self.critical_paths
        }

    async def generate_summary(self, content: str) -> str:
        tree = ast.parse(content)
        summary = {
            "imports": [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)],
            "functions": [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)],
            "classes": [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        }
        return await self.llm_api.analyze(str(summary), "Generate a high-level summary of this code structure.")

    def build_dependency_graph(self, content: str):
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.dependency_graph.add_node(node.name)
                for child_node in ast.walk(node):
                    if isinstance(child_node, ast.Name) and isinstance(child_node.ctx, ast.Load):
                        if child_node.id in self.dependency_graph:
                            self.dependency_graph.add_edge(node.name, child_node.id)

    def split_code(self, content: str) -> List[Dict[str, Any]]:
        tree = ast.parse(content)
        blocks = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                block = ast.get_source_segment(content, node)
                if block:
                    blocks.append({
                        "name": node.name,
                        "type": type(node).__name__,
                        "code": block
                    })
        return blocks

    def identify_entry_point(self, content: str) -> str:
        main_block_start = content.find('if __name__ == "__main__":')
        if main_block_start != -1:
            main_block_end = content.find('\n\n', main_block_start)
            return content[main_block_start:main_block_end if main_block_end != -1 else None]
        return ""

    async def analyze_entry_point(self, entry_point: str) -> Dict[str, Any]:
        if not entry_point:
            return {"analysis": "No clear entry point found in the code."}
        
        prompt = f"""
        Analyze the following entry point of the Python program:

        {entry_point}

        Focus on:
        1. The main function or code block that is executed
        2. Any command-line arguments or configuration being processed
        3. The sequence of operations or function calls in the main execution flow
        4. Any setup or initialization procedures
        5. The overall purpose or functionality initiated from this entry point

        Provide a concise analysis of how this entry point sets up and starts the program's execution.
        """
        analysis = await self.llm_api.analyze(entry_point, prompt)
        return {"code": entry_point, "analysis": analysis}

    def identify_critical_paths(self, entry_point: str):
        tree = ast.parse(entry_point)
        main_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        
        for call in main_calls:
            if isinstance(call.func, ast.Name):
                start_node = call.func.id
                path = self.trace_path(start_node)
                if path:
                    self.critical_paths.append(path)

    def trace_path(self, start_node: str) -> List[str]:
        path = []
        visited = set()
        queue = deque([(start_node, [start_node])])
        
        while queue:
            current, current_path = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            path = current_path
            
            if current not in self.dependency_graph:
                break
            
            for neighbor in self.dependency_graph.successors(current):
                if neighbor not in visited:
                    queue.append((neighbor, current_path + [neighbor]))
        
        return path

    async def analyze_block_with_context(self, block: Dict[str, Any], summary: str, block_index: int) -> Dict[str, Any]:
        local_context = self.get_block_context(block['name'])
        compressed_context = self.compress_context(local_context)
        relevant_global_context = self.get_relevant_global_context()
        relevant_indexed_info = self.get_relevant_indexed_info(block['code'])
        relevant_anchors = self.get_relevant_anchors(block['name'])
        critical_path_info = self.get_critical_path_info(block['name'])
        
        prompt = f"""
        Analyze this code block in the context of the following information:

        Overall Summary:
        {summary}

        Block Type: {block['type']}
        Block Name: {block['name']}

        Local Context:
        {compressed_context}

        Relevant Global Context:
        {relevant_global_context}

        Relevant Indexed Information:
        {relevant_indexed_info}

        Relevant Context Anchors:
        {relevant_anchors}

        Critical Path Information:
        {critical_path_info}

        Code block:
        {block['code']}

        Provide a detailed analysis of this block, focusing on:
        1. Its specific role and functionality
        2. How it relates to its immediate dependencies and dependents
        3. Its place in the overall structure (global context)
        4. Its relationship to the identified context anchors
        5. Its role in the critical execution paths of the program
        6. Any notable patterns or potential issues
        7. How it relates to the indexed information provided

        When referencing global context, indexed information, context anchors, or critical paths, be explicit about how this current block relates to or differs from them.
        Pay special attention to the block's role in any critical paths it's part of, and how this impacts the overall program flow.
        Limit your analysis to the provided contexts and avoid speculating about parts of the code not mentioned here.
        """
        analysis = await self.llm_api.analyze(block['code'], prompt)
        return {"name": block['name'], "type": block['type'], "code": block['code'], "analysis": analysis}

    def get_block_context(self, block_name: str) -> str:
        return json.dumps(self.local_contexts.get(block_name, {}))

    def compress_context(self, context: str) -> str:
        # 使用TF-IDF压缩上下文
        if not hasattr(self, 'tfidf_matrix'):
            self.tfidf_matrix = self.indexer.tfidf_vectorizer.fit_transform([context])
        else:
            self.tfidf_matrix = self.indexer.tfidf_vectorizer.transform([context])
        
        feature_names = self.indexer.tfidf_vectorizer.get_feature_names_out()
        important_words = [feature_names[i] for i in self.tfidf_matrix.indices]
        
        # 只保留TF-IDF值最高的前20个词
        compressed_context = ' '.join(important_words[:20])
        return compressed_context

    def get_relevant_global_context(self) -> str:
        return json.dumps(self.global_context)

    def get_relevant_indexed_info(self, code: str) -> str:
        search_results = self.indexer.search(code)
        return json.dumps([{"name": r['name'], "type": r['type']} for r in search_results])

    def get_relevant_anchors(self, block_name: str) -> str:
        relevant_anchors = []
        for anchor in self.context_anchors:
            if (anchor.name in self.dependency_graph.predecessors(block_name) or
                anchor.name in self.dependency_graph.successors(block_name)):
                relevant_anchors.append(f"{anchor.name} ({anchor.type}): {anchor.summary}")
        return "\n".join(relevant_anchors)

    def get_critical_path_info(self, block_name: str) -> str:
        paths_info = []
        for i, path in enumerate(self.critical_paths):
            if block_name in path:
                position = path.index(block_name)
                prev = path[position - 1] if position > 0 else "Start"
                next = path[position + 1] if position < len(path) - 1 else "End"
                paths_info.append(f"Path {i + 1}: ... -> {prev} -> {block_name} -> {next} -> ...")
        
        if paths_info:
            return "This block is part of the following critical paths:\n" + "\n".join(paths_info)
        else:
            return "This block is not part of any identified critical paths."

    def save_context_snapshot(self, block_index: int, block_analysis: Dict[str, Any]):
        self.context_snapshots.append(block_analysis)

    async def update_global_context(self, block_analysis: Dict[str, Any]):
        self.global_context[block_analysis['name']] = block_analysis['analysis'][:200]

    def update_local_context(self, block_name: str, block_analysis: Dict[str, Any]):
        if block_name not in self.local_contexts:
            self.local_contexts[block_name] = {}
        self.local_contexts[block_name].update({
            "analysis": block_analysis['analysis'][:200],
            "type": block_analysis['type'],
            "dependencies": list(self.dependency_graph.predecessors(block_name)),
            "dependents": list(self.dependency_graph.successors(block_name))
        })

    def should_restructure_context(self) -> bool:
        total_context_size = (
            len(json.dumps(self.global_context)) +
            sum(len(json.dumps(ctx)) for ctx in self.local_contexts.values())
        )
        return total_context_size > self.context_size_threshold

    async def restructure_context(self):
        global_context_str = json.dumps(self.global_context)
        restructured_global = await self.llm_api.analyze(global_context_str, """
        Restructure and consolidate this global context information. 
        Focus on key components, their main purposes, and critical relationships. 
        The restructured context should be about 50% of the original length.
        """)
        self.global_context = {"restructured_global": restructured_global}

        for block_name, local_context in self.local_contexts.items():
            local_context_str = json.dumps(local_context)
            restructured_local = await self.llm_api.analyze(local_context_str, f"""
            Restructure and consolidate the local context for the block '{block_name}'.
            Focus on the most important information about this block's functionality and relationships.
            The restructured context should be about 50% of the original length.
            """)
            self.local_contexts[block_name] = {"restructured_local": restructured_local}

    def is_context_anchor(self, block_analysis: Dict[str, Any]) -> bool:
        return (block_analysis['type'] == 'ClassDef' or 
                (block_analysis['type'] == 'FunctionDef' and len(self.dependency_graph.edges(block_analysis['name'])) > 3))

    def create_context_anchor(self, block_analysis: Dict[str, Any]) -> ContextAnchor:
        importance = self.calculate_anchor_importance(block_analysis)
        return ContextAnchor(
            name=block_analysis['name'],
            type=block_analysis['type'],
            importance=importance,
            summary=block_analysis['analysis'][:200]
        )

    def calculate_anchor_importance(self, block_analysis: Dict[str, Any]) -> float:
        return len(self.dependency_graph.edges(block_analysis['name'])) * 0.1

    async def generate_project_summary(self, file_summary: str, block_analyses: List[Dict[str, Any]]) -> str:
        context = {
            "file_summary": file_summary,
            "block_summaries": [{"name": ba['name'], "type": ba['type'], "summary": ba['analysis'][:200]} for ba in block_analyses],
            "dependency_graph": nx.node_link_data(self.dependency_graph)
        }
        prompt = """
        Based on the provided information, generate a comprehensive project summary that includes:
        1. An overview of the project's purpose and main functionality
        2. Key components and their roles
        3. Important relationships and dependencies between components
        4. Main algorithms or processes implemented
        5. Notable design patterns or architectural choices
        6. Potential areas of complexity or importance

        This summary should serve as a high-level guide to understanding the project structure and functionality.
        """
        return await self.llm_api.analyze(json.dumps(context), prompt)

    async def integrate_analyses(self, summary: str, block_analyses: List[Dict[str, Any]], entry_point_analysis: Dict[str, Any]) -> str:
        context = {
            "summary": summary,
            "project_summary": self.project_summary,
            "entry_point_analysis": entry_point_analysis,
            "global_context": self.global_context,
            "critical_paths": self.critical_paths,
            "context_anchors": [vars(anchor) for anchor in self.context_anchors]
        }
        prompt = """
        Provide an integrated analysis of the entire codebase based on the following information:
        
        1. Overall summary
        2. Project summary
        3. Entry point analysis
        4. Global context
        5. Critical execution paths
        6. Context anchors

        Focus on:
        - The overall structure and architecture of the code
        - The program's entry point and how it initiates the main functionality
        - Key components and their roles, especially the identified context anchors
        - How different parts of the code interact with each other
        - The flow of execution from the entry point through the main components, particularly along critical paths
        - Any patterns or design principles evident in the code structure
        - Potential areas for improvement or refactoring
        - The significance of the identified critical paths and their impact on the program's functionality

        Synthesize all this information to give a comprehensive understanding of the codebase, 
        highlighting how the different pieces fit together and any notable transitions or developments in the code structure.
        Use the project summary, entry point analysis, and context anchors to provide a high-level perspective on the codebase.
        Pay special attention to how the critical paths reveal the core functionality and execution flow of the program.
        """
        return await self.llm_api.analyze(json.dumps(context), prompt)
    
import openai
import asyncio
import json
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_random_exponential

class EnhancedLLMApi:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key
        self.conversation_history = []
        self.max_history_length = 10
        self.token_limit = 8000  # Adjust based on the model's actual limit

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def analyze(self, content: str, prompt: str) -> str:
        # Prepare the messages
        messages = self._prepare_messages(content, prompt)
        
        # Make the API call
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0.5,
            )
            analysis = response.choices[0].message['content'].strip()
            
            # Update conversation history
            self._update_conversation_history(prompt, analysis)
            
            return analysis
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            raise

    def _prepare_messages(self, content: str, prompt: str) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in analyzing Python code."},
            {"role": "user", "content": f"Here's the code or context to analyze:\n\n{content}\n\nAnalysis prompt: {prompt}"}
        ]
        
        # Add relevant conversation history
        messages.extend(self.conversation_history)
        
        return messages

    def _update_conversation_history(self, prompt: str, response: str):
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Limit the history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    async def analyze_with_context(self, content: str, prompt: str, context: Dict[str, Any]) -> str:
        context_str = json.dumps(context)
        full_prompt = f"{prompt}\n\nAdditional context: {context_str}"
        return await self.analyze(content, full_prompt)

    async def summarize(self, text: str, max_length: int = 200) -> str:
        prompt = f"Summarize the following text in about {max_length} words:\n\n{text}"
        return await self.analyze("", prompt)

    async def compare_code_blocks(self, block1: str, block2: str) -> str:
        prompt = f"Compare and contrast the following two code blocks:\n\nBlock 1:\n{block1}\n\nBlock 2:\n{block2}\n\nFocus on their functionality, structure, and any notable differences or similarities."
        return await self.analyze("", prompt)

    async def identify_design_patterns(self, code: str) -> List[str]:
        prompt = "Identify any design patterns used in the following code. List the patterns and briefly explain how they are implemented:"
        analysis = await self.analyze(code, prompt)
        # This is a simple extraction. In a real scenario, you might want to use more sophisticated NLP techniques.
        patterns = [line.split(':')[0] for line in analysis.split('\n') if ':' in line]
        return patterns

    async def suggest_improvements(self, code: str) -> str:
        prompt = "Analyze the following code and suggest potential improvements in terms of efficiency, readability, and best practices:"
        return await self.analyze(code, prompt)

    async def explain_code_section(self, code: str, section_name: str) -> str:
        prompt = f"Explain the purpose and functionality of the '{section_name}' section in the following code:"
        return await self.analyze(code, prompt)

    async def analyze_complexity(self, code: str) -> Dict[str, Any]:
        prompt = "Analyze the complexity of the following code. Consider time complexity, space complexity, and cognitive complexity. Provide a brief explanation for each:"
        analysis = await self.analyze(code, prompt)
        
        # This is a simple parsing. In a real scenario, you might want to use more sophisticated techniques.
        complexity = {}
        current_key = ""
        for line in analysis.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                current_key = key.strip().lower().replace(' ', '_')
                complexity[current_key] = value.strip()
            elif current_key:
                complexity[current_key] += ' ' + line.strip()
        
        return complexity

    async def generate_unit_tests(self, code: str) -> str:
        prompt = "Generate unit tests for the following code. Include test cases for normal operation, edge cases, and potential error conditions:"
        return await self.analyze(code, prompt)

    async def analyze_security(self, code: str) -> List[Dict[str, str]]:
        prompt = "Analyze the following code for potential security vulnerabilities. Identify any issues and suggest mitigations:"
        analysis = await self.analyze(code, prompt)
        
        # Simple parsing of the analysis. In a real scenario, you might want to use more sophisticated techniques.
        vulnerabilities = []
        current_vuln = {}
        for line in analysis.split('\n'):
            if line.startswith('Vulnerability:'):
                if current_vuln:
                    vulnerabilities.append(current_vuln)
                current_vuln = {"type": line.split(':', 1)[1].strip()}
            elif line.startswith('Mitigation:'):
                current_vuln["mitigation"] = line.split(':', 1)[1].strip()
        if current_vuln:
            vulnerabilities.append(current_vuln)
        
        return vulnerabilities

# Usage example
async def main():
    api_key = "your-openai-api-key-here"
    llm_api = EnhancedLLMApi(api_key)
    
    code = """
    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    """
    
    analysis = await llm_api.analyze(code, "Explain this function and suggest any improvements.")
    print("Analysis:", analysis)
    
    summary = await llm_api.summarize(code)
    print("Summary:", summary)
    
    patterns = await llm_api.identify_design_patterns(code)
    print("Design Patterns:", patterns)
    
    complexity = await llm_api.analyze_complexity(code)
    print("Complexity Analysis:", complexity)
    
    security_analysis = await llm_api.analyze_security(code)
    print("Security Analysis:", security_analysis)

if __name__ == "__main__":
    asyncio.run(main())
    

# 使用示例
async def main():
    api_key = "your-api-key-here"
    llm_api = EnhancedLLMApi(api_key)  # 假设您有一个 EnhancedLLMApi 类
    
    large_file_path = "/path/to/your/large/file.py"
    analyzer = LargeCodeAnalyzer(llm_api)
    analysis_result = await analyzer.analyze_large_file(large_file_path)
    
    print("Project Summary:")
    print(analysis_result['project_summary'])
    
    print("\nEntry Point Analysis:")
    print(analysis_result['entry_point_analysis']['analysis'])
    
    print("\nContext Anchors:")
    for anchor in analysis_result['context_anchors']:
        print(f"{anchor['name']} ({anchor['type']}) - Importance: {anchor['importance']}")
        print(f"Summary: {anchor['summary']}")
        print()
    
    print("\nCritical Paths:")
    for i, path in enumerate(analysis_result['critical_paths']):
        print(f"Path {i + 1}: {' -> '.join(path)}")
    
    print("\nFinal Integrated Analysis:")
    print(analysis_result['final_analysis'])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

