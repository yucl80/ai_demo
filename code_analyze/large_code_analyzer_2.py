import ast
import json
import networkx as nx
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from collections import deque

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
        self.context_size_threshold = 5000
        self.block_summaries = {}
        self.indexer = CodeIndexer()
        self.project_summary = ""
        self.context_anchors = []
        self.critical_paths = []
        self.local_contexts = {}

    async def analyze_large_file(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as file:
            content = file.read()
        
        summary = await self.generate_summary(content)
        self.build_dependency_graph(content)
        code_blocks = self.split_code(content)
        
        # 识别程序入口点
        entry_point = self.identify_entry_point(content)
        entry_point_analysis = await self.analyze_entry_point(entry_point)
        
        # 识别关键路径
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
            
             # 识别并添加上下文锚点
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
            "block_analyses": block_analyses,
            "final_analysis": final_analysis,
            "dependency_graph": nx.node_link_data(self.dependency_graph),
            "global_context": self.global_context,
            "index": self.indexer.index,
            "context_anchors": [vars(anchor) for anchor in self.context_anchors],
            "critical_paths": self.critical_paths
        }
        
    def should_restructure_context(self) -> bool:
        total_context_size = (
            len(json.dumps(self.global_context)) +
            sum(len(json.dumps(ctx)) for ctx in self.local_contexts.values())
        )
        return total_context_size > self.context_size_threshold
    

    async def restructure_context(self):
        # 重构全局上下文
        global_context_str = json.dumps(self.global_context)
        restructured_global = await self.llm_api.analyze(global_context_str, """
        Restructure and consolidate this global context information. 
        Focus on key components, their main purposes, and critical relationships. 
        The restructured context should be about 50% of the original length.
        """)
        self.global_context = {"restructured_global": restructured_global}

        # 重构局部上下文
        for block_name, local_context in self.local_contexts.items():
            local_context_str = json.dumps(local_context)
            restructured_local = await self.llm_api.analyze(local_context_str, f"""
            Restructure and consolidate the local context for the block '{block_name}'.
            Focus on the most important information about this block's functionality and relationships.
            The restructured context should be about 50% of the original length.
            """)
            self.local_contexts[block_name] = {"restructured_local": restructured_local}

    def update_local_context(self, block_name: str, block_analysis: Dict[str, Any]):
        if block_name not in self.local_contexts:
            self.local_contexts[block_name] = {}
        self.local_contexts[block_name].update({
            "analysis": block_analysis['analysis'][:200],
            "type": block_analysis['type'],
            "dependencies": list(self.dependency_graph.predecessors(block_name)),
            "dependents": list(self.dependency_graph.successors(block_name))
        })

    
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

        
    def identify_entry_point(self, content: str) -> str:
        # 使用AST或正则表达式来查找 if __name__ == "__main__": 语句
        # 这里使用一个简单的字符串搜索作为示例
        main_block_start = content.find('if __name__ == "__main__":')
        if main_block_start != -1:
            # 提取main块的内容
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
        
        
    def is_context_anchor(self, block_analysis: Dict[str, Any]) -> bool:
        # 这里可以实现更复杂的逻辑来判断一个代码块是否应该成为上下文锚点
        # 例如，基于代码块的复杂度、依赖关系数量、或者特定的模式
        return (block_analysis['type'] == 'class' or 
                block_analysis['type'] == 'function' and len(self.dependency_graph.edges(block_analysis['name'])) > 3)

    def create_context_anchor(self, block_analysis: Dict[str, Any]) -> ContextAnchor:
        importance = self.calculate_anchor_importance(block_analysis)
        return ContextAnchor(
            name=block_analysis['name'],
            type=block_analysis['type'],
            importance=importance,
            summary=block_analysis['analysis'][:200]
        )

    def calculate_anchor_importance(self, block_analysis: Dict[str, Any]) -> float:
        # 这里可以实现更复杂的逻辑来计算锚点的重要性
        # 例如，基于依赖关系的数量、代码复杂度等
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
    
    def get_block_context(self, block_name: str) -> str:
        return json.dumps(self.local_contexts.get(block_name, {}))

    def get_relevant_global_context(self) -> str:
        return json.dumps(self.global_context)

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
        
    async def analyze_block_with_context(self, block: Dict[str, Any], summary: str, block_index: int) -> Dict[str, Any]:
        local_context = self.get_block_context(block['name'])
        compressed_context = self.compress_context(local_context)
        relevant_global_context = self.get_relevant_global_context()
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

        Critical Path Information:
        {critical_path_info}

        Code block:
        {block['code']}

        Provide a detailed analysis of this block, focusing on:
        1. Its specific role and functionality
        2. How it relates to its immediate dependencies and dependents
        3. Its place in the overall structure (global context)
        4. Its role in the critical execution paths of the program
        5. Any notable patterns or potential issues

        When referencing global context or critical paths, be explicit about how this current block relates to or differs from them.
        Pay special attention to the block's role in any critical paths it's part of, and how this impacts the overall program flow.
        Limit your analysis to the provided contexts and avoid speculating about parts of the code not mentioned here.
        """
        analysis = await self.llm_api.analyze(block['code'], prompt)
        return {"name": block['name'], "type": block['type'], "code": block['code'], "analysis": analysis}


    def get_relevant_anchors(self, block_name: str) -> str:
        relevant_anchors = []
        for anchor in self.context_anchors:
            if (anchor.name in self.dependency_graph.predecessors(block_name) or
                anchor.name in self.dependency_graph.successors(block_name)):
                relevant_anchors.append(f"{anchor.name} ({anchor.type}): {anchor.summary}")
        return "\n".join(relevant_anchors)

    async def integrate_analyses(self, summary: str, block_analyses: List[Dict[str, Any]], entry_point_analysis: Dict[str, Any]) -> str:
        context = {
            "summary": summary,
            "project_summary": self.project_summary,
            "entry_point_analysis": entry_point_analysis,
            "block_analyses": [{"name": ba['name'], "type": ba['type'], "analysis": ba['analysis']} for ba in block_analyses],
            "dependency_graph": nx.node_link_data(self.dependency_graph),
            "global_context": self.global_context,
            "index": self.indexer.index,
            "context_anchors": [vars(anchor) for anchor in self.context_anchors],
            "critical_paths": self.critical_paths
        }
        prompt = """
        Provide an integrated analysis of the entire codebase based on the following information:
        
        1. Overall summary
        2. Project summary
        3. Entry point analysis
        4. Individual block analyses
        5. Dependency relationships between different parts of the code
        6. Global context
        7. Code index
        8. Context anchors
        9. Critical execution paths

        Focus on:
        - The overall structure and architecture of the code
        - The program's entry point and how it initiates the main functionality
        - Key components and their roles, especially the identified context anchors
        - How different parts of the code interact with each other
        - The flow of execution from the entry point through the main components, particularly along critical paths
        - Any patterns or design principles evident in the code structure
        - How the understanding of the code evolved as different blocks were analyzed
        - Potential areas for improvement or refactoring
        - How the indexed information and context anchors contribute to understanding the codebase
        - The significance of the identified critical paths and their impact on the program's functionality

        Synthesize all this information to give a comprehensive understanding of the codebase, 
        highlighting how the different pieces fit together and any notable transitions or developments in the code structure.
        Use the project summary, entry point analysis, index, context anchors, and critical paths to provide a high-level perspective on the codebase.
        Pay special attention to how the critical paths reveal the core functionality and execution flow of the program.
        """
        return await self.llm_api.analyze(json.dumps(context), prompt)
    
    async def integrate_analyses2(self, summary: str, block_analyses: List[Dict[str, Any]], entry_point_analysis: Dict[str, Any]) -> str:
        context = {
            "summary": summary,
            "project_summary": self.project_summary,
            "entry_point_analysis": entry_point_analysis,
            "global_context": self.global_context,
            "critical_paths": self.critical_paths
        }
        prompt = """
        Provide an integrated analysis of the entire codebase based on the following information:
        
        1. Overall summary
        2. Project summary
        3. Entry point analysis
        4. Global context
        5. Critical execution paths

        Focus on:
        - The overall structure and architecture of the code
        - The program's entry point and how it initiates the main functionality
        - Key components and their roles
        - How different parts of the code interact with each other
        - The flow of execution from the entry point through the main components, particularly along critical paths
        - Any patterns or design principles evident in the code structure
        - Potential areas for improvement or refactoring
        - The significance of the identified critical paths and their impact on the program's functionality

        Synthesize all this information to give a comprehensive understanding of the codebase, 
        highlighting how the different pieces fit together and any notable transitions or developments in the code structure.
        Pay special attention to how the critical paths reveal the core functionality and execution flow of the program.
        """
        return await self.llm_api.analyze(json.dumps(context), prompt)



    
   
    
    
# 使用示例
async def main():
    api_key = "your-api-key-here"
    llm_api = EnhancedLLMApi(api_key)
    
    large_file_path = "/path/to/your/large/file.py"
    analyzer = LargeCodeAnalyzer(llm_api)
    analysis_result = await analyzer.analyze_large_file(large_file_path)
    
    print("Project Summary:")
    print(analysis_result['project_summary'])
    
    print("\nContext Anchors:")
    for anchor in analysis_result['context_anchors']:
        print(f"{anchor['name']} ({anchor['type']}) - Importance: {anchor['importance']}")
        print(f"Summary: {anchor['summary']}")
        print()
    
    print("\nFinal Integrated Analysis:")
    print(analysis_result['final_analysis'])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

  
