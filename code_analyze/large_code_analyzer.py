import ast
import json
import networkx as nx
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from collections import deque

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
        self.context_size_threshold = 1000
        self.block_summaries = {}
        self.indexer = CodeIndexer()
        self.project_summary = ""

    async def analyze_large_file(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as file:
            content = file.read()
        
        summary = await self.generate_summary(content)
        self.build_dependency_graph(content)
        code_blocks = self.split_code(content)
        
        block_analyses = []
        for i, block in enumerate(code_blocks):
            block_analysis = await self.analyze_block_with_context(block, summary, i)
            block_analyses.append(block_analysis)
            self.save_context_snapshot(i, block_analysis)
            await self.update_global_context(block_analysis)
            self.block_summaries[block['name']] = block_analysis['analysis'][:200]
            self.indexer.add_to_index(block['name'], block['code'], block['type'])
        
        self.project_summary = await self.generate_project_summary(summary, block_analyses)
        final_analysis = await self.integrate_analyses(summary, block_analyses)
        
        return {
            "summary": summary,
            "project_summary": self.project_summary,
            "block_analyses": block_analyses,
            "final_analysis": final_analysis,
            "dependency_graph": nx.node_link_data(self.dependency_graph),
            "global_context": self.global_context,
            "index": self.indexer.index
        }

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

    async def analyze_block_with_context(self, block: Dict[str, Any], summary: str, block_index: int) -> Dict[str, Any]:
        local_context = self.get_block_context(block['name'])
        compressed_context = self.compress_context(local_context)
        relevant_previous_contexts = self.get_relevant_previous_contexts(block['name'])
        relevant_global_context = self.get_relevant_global_context(block['name'])
        relevant_indexed_info = self.get_relevant_indexed_info(block['code'])
        
        prompt = f"""
        Analyze this code block in the context of the following information:

        Overall Summary:
        {summary}

        Block Type: {block['type']}
        Block Name: {block['name']}

        Local Context:
        {compressed_context}

        Relevant Previous Contexts:
        {relevant_previous_contexts}

        Relevant Global Context:
        {relevant_global_context}

        Relevant Indexed Information:
        {relevant_indexed_info}

        Code block:
        {block['code']}

        Provide a detailed analysis of this block, focusing on:
        1. Its specific role and functionality
        2. How it relates to its immediate dependencies (local context)
        3. Its place in the overall structure (global context)
        4. How it connects to or builds upon previously analyzed relevant blocks
        5. Any notable patterns or potential issues
        6. How it relates to the indexed information provided

        When referencing previous blocks, global context, or indexed information, be explicit about how this current block relates to or differs from them.
        Limit your analysis to the provided contexts and avoid speculating about parts of the code not mentioned here.
        """
        analysis = await self.llm_api.analyze(block['code'], prompt)
        return {"name": block['name'], "type": block['type'], "code": block['code'], "analysis": analysis}

    def get_relevant_indexed_info(self, code: str) -> str:
        search_results = self.indexer.search(code)
        return json.dumps([{"name": r['name'], "type": r['type']} for r in search_results])



   

    def get_block_context(self, block_name: str) -> str:
        predecessors = list(self.dependency_graph.predecessors(block_name))
        successors = list(self.dependency_graph.successors(block_name))
        context = f"This {block_name} is used by: {', '.join(predecessors)}\n" if predecessors else ""
        context += f"This {block_name} uses: {', '.join(successors)}\n" if successors else ""
        return context
        
    
    def compress_context(self, context: str) -> str:
        # 使用TF-IDF压缩上下文
        if not hasattr(self, 'tfidf_matrix'):
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([context])
        else:
            self.tfidf_matrix = self.tfidf_vectorizer.transform([context])
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        important_words = [feature_names[i] for i in self.tfidf_matrix.indices]
        
        # 只保留TF-IDF值最高的前10个词
        compressed_context = ' '.join(important_words[:10])
        return compressed_context

    def get_previous_context(self, current_block_index: int) -> str:
        if not self.context_snapshots:
            return "No previous context available."
        previous_analysis = self.context_snapshots[-1]
        return f"Previous block ({previous_analysis['name']}) analysis summary: {previous_analysis['analysis'][:200]}..."

    def save_context_snapshot(self, block_index: int, block_analysis: Dict[str, Any]):
        self.context_snapshots.append(block_analysis)

    async def update_global_context(self, block_analysis: Dict[str, Any]):
        self.global_context[block_analysis['name']] = block_analysis['analysis'][:200]  # 保存简短摘要
        
        if len(json.dumps(self.global_context)) > self.context_size_threshold:
            await self.restructure_global_context()

    def save_context_snapshot(self, block_index: int, block_analysis: Dict[str, Any]):
        self.context_snapshots.append(block_analysis)
        
        # 如果快照数量过多，可以考虑只保留最近的几个
        if len(self.context_snapshots) > 5:
            self.context_snapshots.pop(0)
        
        # 将快照保存到文件（可选）
        with open(f'context_snapshot_{block_index}.pkl', 'wb') as f:
            pickle.dump(block_analysis, f)

    def load_context_snapshot(self, block_index: int) -> Dict[str, Any]:
        # 从文件加载快照（如果需要）
        with open(f'context_snapshot_{block_index}.pkl', 'rb') as f:
            return pickle.load(f)

    async def update_global_context(self, block_analysis: Dict[str, Any]):
        self.global_context[block_analysis['name']] = block_analysis['analysis'][:200]  # 保存简短摘要
        
        if len(json.dumps(self.global_context)) > self.context_size_threshold:
            await self.restructure_global_context()
            
    async def analyze_block_with_context(self, block: Dict[str, Any], summary: str, block_index: int) -> Dict[str, Any]:
        local_context = self.get_block_context(block['name'])
        compressed_context = self.compress_context(local_context)
        relevant_previous_contexts = self.get_relevant_previous_contexts(block['name'])
        relevant_global_context = self.get_relevant_global_context(block['name'])
        
        prompt = f"""
        Analyze this code block in the context of the following information:

        Overall Summary:
        {summary}

        Block Type: {block['type']}
        Block Name: {block['name']}

        Local Context:
        {compressed_context}

        Relevant Previous Contexts:
        {relevant_previous_contexts}

        Relevant Global Context:
        {relevant_global_context}

        Code block:
        {block['code']}

        Provide a detailed analysis of this block, focusing on:
        1. Its specific role and functionality
        2. How it relates to its immediate dependencies (local context)
        3. Its place in the overall structure (global context)
        4. How it connects to or builds upon previously analyzed relevant blocks
        5. Any notable patterns or potential issues

        When referencing previous blocks or global context, be explicit about how this current block relates to or differs from them.
        Limit your analysis to the provided contexts and avoid speculating about parts of the code not mentioned here.
        """
        analysis = await self.llm_api.analyze(block['code'], prompt)
        return {"name": block['name'], "type": block['type'], "code": block['code'], "analysis": analysis}

    def get_relevant_previous_contexts(self, block_name: str) -> str:
        relevant_contexts = []
        for predecessor in self.dependency_graph.predecessors(block_name):
            if predecessor in self.block_summaries:
                relevant_contexts.append(f"{predecessor}: {self.block_summaries[predecessor]}")
        
        # 也考虑最近分析的几个块
        recent_blocks = list(self.context_snapshots)[-3:]  # 获取最近的3个块
        for block in recent_blocks:
            if block['name'] != block_name and block['name'] not in relevant_contexts:
                relevant_contexts.append(f"{block['name']}: {block['analysis'][:200]}")
        
        return "\n".join(relevant_contexts)

    def get_relevant_global_context(self, block_name: str) -> str:
        if "restructured_context" in self.global_context:
            return self.global_context["restructured_context"]
        
        relevant_context = {}
        for name, summary in self.global_context.items():
            if name == block_name or name in self.dependency_graph.predecessors(block_name) or name in self.dependency_graph.successors(block_name):
                relevant_context[name] = summary
        
        return json.dumps(relevant_context)

    async def update_global_context(self, block_analysis: Dict[str, Any]):
        self.global_context[block_analysis['name']] = block_analysis['analysis'][:200]
        
        if len(json.dumps(self.global_context)) > self.context_size_threshold:
            await self.restructure_global_context()

    async def restructure_global_context(self):
        context_str = json.dumps(self.global_context)
        prompt = f"""
        The following is a collection of summaries for different parts of a large codebase:

        {context_str}

        Please restructure and consolidate this information into a more concise global context. 
        Focus on key components, their main purposes, and critical relationships. 
        Highlight how different parts of the code relate to each other and any overarching patterns or principles.
        The restructured context should be about 50% of the original length.
        """
        restructured_context = await self.llm_api.analyze(context_str, prompt)
        self.global_context = {"restructured_context": restructured_context}

    async def integrate_analyses(self, summary: str, block_analyses: List[Dict[str, Any]]) -> str:
        context = {
            "summary": summary,
            "project_summary": self.project_summary,
            "block_analyses": [{"name": ba['name'], "type": ba['type'], "analysis": ba['analysis']} for ba in block_analyses],
            "dependency_graph": nx.node_link_data(self.dependency_graph),
            "global_context": self.global_context,
            "index": self.indexer.index
        }
        prompt = """
        Provide an integrated analysis of the entire codebase based on the following information:
        
        1. Overall summary
        2. Project summary
        3. Individual block analyses
        4. Dependency relationships between different parts of the code
        5. Global context
        6. Code index

        Focus on:
        - The overall structure and architecture of the code
        - Key components and their roles
        - How different parts of the code interact with each other
        - Any patterns or design principles evident in the code structure
        - How the understanding of the code evolved as different blocks were analyzed
        - Potential areas for improvement or refactoring
        - How the indexed information contributes to understanding the codebase

        Synthesize all this information to give a comprehensive understanding of the codebase, 
        highlighting how the different pieces fit together and any notable transitions or developments in the code structure.
        Use the project summary and index to provide a high-level perspective on the codebase.
        """
        return await self.llm_api.analyze(json.dumps(context), prompt)
   
    
    
async def main():
    api_key = "your-api-key-here"
    llm_api = EnhancedLLMApi(api_key)
    
    large_file_path = "/path/to/your/large/file.py"
    analyzer = LargeCodeAnalyzer(llm_api)
    analysis_result = await analyzer.analyze_large_file(large_file_path)
    
    print("Project Summary:")
    print(analysis_result['project_summary'])
    
    print("\nBlock Analyses:")
    for block_analysis in analysis_result['block_analyses']:
        print(f"\n{block_analysis['type']} {block_analysis['name']}:")
        print(block_analysis['analysis'])
    
    print("\nFinal Integrated Analysis:")
    print(analysis_result['final_analysis'])
    
    print("\nCode Index:")
    print(json.dumps(analysis_result['index'], indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

  
