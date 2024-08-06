import argparse
import os
import ast
import asyncio
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import re
import json
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from graphviz import Digraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CodeBlock:
    id: str
    content: str
    comments: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    api_calls: List[str] = field(default_factory=list)
    db_operations: List[str] = field(default_factory=list)
    control_structures: List[str] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    complexity: int = 0
    token_count: int = 0

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import json
from typing import List, Dict, Any

class LLMApi:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as an environment variable or pass it to the constructor.")
        openai.api_key = self.api_key
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            raise

    async def analyze(self, content: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert code analyzer and business process consultant with years of experience in software architecture and business analysis. Your task is to analyze code and extract meaningful business processes and logic."},
            {"role": "user", "content": f"I need you to analyze the following code or content and {prompt}. Please provide a detailed, structured analysis focusing on business processes, logic flows, and any important patterns or practices you identify. Be specific and use examples from the code where relevant.\n\nContent to analyze:\n{content}"}
        ]
        return await self._call_openai_api(messages)

    async def generate_flowchart(self, description: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert in creating detailed, accurate flowcharts from textual descriptions of business processes and code logic. Your task is to convert complex descriptions into clear, structured flowchart representations."},
            {"role": "user", "content": f"Based on the following description, create a detailed flowchart. Provide the flowchart as a series of steps in a structured format that can be easily converted to a visual representation. Use proper flowchart symbols (e.g., process, decision, input/output) and ensure the logic flow is clear and accurate.\n\nDescription:\n{description}"}
        ]
        return await self._call_openai_api(messages)

    async def extract_business_logic(self, code: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert in extracting and interpreting business logic from source code. Your task is to analyze code and identify the key business rules, processes, and logic embedded within it."},
            {"role": "user", "content": f"Examine the following code and extract the core business logic. Focus on identifying business rules, key processes, data transformations, and decision points. Explain each identified piece of business logic in plain language, relating it to potential real-world business operations or requirements.\n\nCode to analyze:\n{code}"}
        ]
        return await self._call_openai_api(messages)

    async def analyze_code_quality(self, pylint_report: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert in code quality analysis with deep knowledge of best practices, design patterns, and common pitfalls in software development. Your task is to interpret static code analysis reports and provide actionable insights."},
            {"role": "user", "content": f"Analyze the following Pylint report and provide a comprehensive assessment of the code quality. Identify major issues, potential risks, and areas for improvement. Prioritize your findings based on their potential impact on code maintainability, performance, and reliability. Offer specific, actionable recommendations for addressing each issue.\n\nPylint report:\n{pylint_report}"}
        ]
        return await self._call_openai_api(messages)

    async def analyze_code_evolution(self, evolution_data: List[Dict[str, Any]]) -> str:
        messages = [
            {"role": "system", "content": "You are an expert in analyzing code evolution and software project dynamics over time. Your task is to interpret historical code metrics and provide insights into the development trends and potential issues in the project's lifecycle."},
            {"role": "user", "content": f"Examine the following code evolution data and provide a detailed analysis of how the codebase has changed over time. Focus on identifying trends in code complexity, size, and composition. Highlight any significant changes or patterns that might indicate improvements or potential issues in the development process. Offer insights into the project's health and suggestions for future development strategies.\n\nEvolution data:\n{json.dumps(evolution_data, indent=2)}"}
        ]
        return await self._call_openai_api(messages)

    async def analyze_critical_path(self, path_content: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert in analyzing critical paths and execution flows in complex software systems. Your task is to examine code execution paths and explain their significance in terms of business operations and system performance."},
            {"role": "user", "content": f"Analyze the following execution path and explain its business and technical significance. Identify key operations, potential bottlenecks, and critical decision points. Relate the technical flow to business processes and explain how this path contributes to the overall system functionality. Suggest any potential optimizations or areas for closer monitoring.\n\nExecution path:\n{path_content}"}
        ]
        return await self._call_openai_api(messages)

    async def summarize_cluster(self, cluster_content: List[str]) -> str:
        combined_content = "\n\n".join(cluster_content)
        messages = [
            {"role": "system", "content": "You are an expert in code analysis and pattern recognition, specializing in identifying common themes and purposes across diverse code segments. Your task is to synthesize information from multiple code blocks and extract overarching concepts and functionalities."},
            {"role": "user", "content": f"Examine the following collection of code blocks and provide a comprehensive summary of their common themes, purposes, and functionalities. Identify any shared patterns, repeated business logic, or related operations across these blocks. Suggest potential opportunities for code consolidation, shared libraries, or architectural improvements based on your analysis.\n\nCode blocks:\n{combined_content}"}
        ]
        return await self._call_openai_api(messages)

    async def interactive_query(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "You are an advanced AI assistant specializing in code analysis and software architecture. Your role is to provide expert insights and answers to questions about codebases, development practices, and software design. Use the provided context to inform your responses, but also draw upon your general knowledge of software engineering best practices."},
            {"role": "user", "content": f"Based on the following context about a codebase, please answer this question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nProvide a detailed, informative answer. If the context doesn't provide enough information to fully answer the question, state that clearly and offer the best possible insight based on the available information and your general knowledge of software development."}
        ]
        return await self._call_openai_api(messages)


class BusinessProcessAnalyzer:
    def __init__(self, llm_api: LLMApi):
        self.llm_api = llm_api
        self.code_blocks: Dict[str, CodeBlock] = {}
        self.call_graph = nx.DiGraph()
        self.data_flow_graph = nx.DiGraph()
        self.domain_knowledge: Dict[str, str] = {}
        self.business_processes: Dict[str, str] = {}

    async def load_code(self, directory: str):
        logger.info(f"Loading code from {directory}")
        tasks = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    tasks.append(self.process_file(file_path))
        await asyncio.gather(*tasks)
        self.build_graphs()
        logger.info("Code loading and graph building completed")

    async def process_file(self, file_path: str):
        logger.info(f"Processing file: {file_path}")
        with open(file_path, 'r') as f:
            content = f.read()
        block_id = os.path.relpath(file_path)
        tree = ast.parse(content)
        
        code_block = CodeBlock(
            id=block_id,
            content=content,
            comments=self.extract_comments(tree),
            functions=self.extract_functions(tree),
            variables=self.extract_variables(tree),
            api_calls=self.extract_api_calls(tree),
            db_operations=self.extract_db_operations(tree),
            control_structures=self.extract_control_structures(tree),
            business_rules=self.extract_business_rules(tree),
            exceptions=self.extract_exceptions(tree),
            imports=self.extract_imports(tree),
            classes=self.extract_classes(tree),
            complexity=self.calculate_complexity(tree),
            token_count=len(word_tokenize(content))
        )
        
        self.code_blocks[block_id] = code_block

    def extract_comments(self, tree: ast.AST) -> List[str]:
        return [node.s for node in ast.walk(tree) if isinstance(node, ast.Str) and isinstance(node.parent, ast.Expr)]

    def extract_functions(self, tree: ast.AST) -> List[str]:
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    def extract_variables(self, tree: ast.AST) -> List[str]:
        return [node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)]

    def extract_api_calls(self, tree: ast.AST) -> List[str]:
        return [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]

    def extract_db_operations(self, tree: ast.AST) -> List[str]:
        db_ops = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ['execute', 'query', 'insert', 'update', 'delete']:
                    db_ops.append(f"{node.func.attr}: {ast.unparse(node.args[0])}")
        return db_ops

    def extract_control_structures(self, tree: ast.AST) -> List[str]:
        control_structs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                control_structs.append(f"{type(node).__name__}: {ast.unparse(node)[:50]}...")
        return control_structs

    def extract_business_rules(self, tree: ast.AST) -> List[str]:
        rules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                condition = ast.unparse(node.test)
                rules.append(f"If {condition}: ...")
            elif isinstance(node, ast.Assert):
                condition = ast.unparse(node.test)
                rules.append(f"Assert {condition}")
        return rules

    def extract_exceptions(self, tree: ast.AST) -> List[str]:
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]

    def extract_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module}.{node.names[0].name}")
        return imports

    def extract_classes(self, tree: ast.AST) -> List[str]:
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    def calculate_complexity(self, tree: ast.AST) -> int:
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1
        return complexity

    def build_graphs(self):
        self.build_call_graph()
        self.build_data_flow_graph()

    def build_call_graph(self):
        for block_id, block in self.code_blocks.items():
            self.call_graph.add_node(block_id)
            for func in block.functions:
                for other_id, other_block in self.code_blocks.items():
                    if func in other_block.content:
                        self.call_graph.add_edge(block_id, other_id)

    def build_data_flow_graph(self):
        for block_id, block in self.code_blocks.items():
            self.data_flow_graph.add_node(block_id)
            for var in block.variables:
                for other_id, other_block in self.code_blocks.items():
                    if var in other_block.content and block_id != other_id:
                        self.data_flow_graph.add_edge(block_id, other_id, variable=var)

    async def analyze_business_processes(self) -> str:
        logger.info("Starting business process analysis")
        overall_analysis = await self.llm_api.analyze(
            self.get_overall_summary(),
            "Provide a high-level summary of the business processes in this codebase"
        )

        detailed_analyses = await self.analyze_blocks_in_parallel()

        flow_analysis = await self.llm_api.analyze(
            json.dumps({
                "call_graph": nx.to_dict_of_dicts(self.call_graph),
                "data_flow_graph": nx.to_dict_of_dicts(self.data_flow_graph)
            }),
            "Describe the overall flow of business processes based on these graphs"
        )

        flowchart = await self.llm_api.generate_flowchart(flow_analysis)

        self.business_processes = await self.extract_business_processes()

        return f"""
        Overall Business Process Analysis:
        {overall_analysis}

        Detailed Analyses of Code Blocks:
        {"".join(detailed_analyses)}

        Business Process Flow Analysis:
        {flow_analysis}

        Business Process Flowchart:
        {flowchart}

        Extracted Business Processes:
        {json.dumps(self.business_processes, indent=2)}
        """

    def get_overall_summary(self) -> str:
        return json.dumps({
            "total_files": len(self.code_blocks),
            "total_functions": sum(len(block.functions) for block in self.code_blocks.values()),
            "total_api_calls": sum(len(block.api_calls) for block in self.code_blocks.values()),
            "total_db_operations": sum(len(block.db_operations) for block in self.code_blocks.values()),
            "total_business_rules": sum(len(block.business_rules) for block in self.code_blocks.values()),
            "total_exceptions": sum(len(block.exceptions) for block in self.code_blocks.values()),
            "average_complexity": sum(block.complexity for block in self.code_blocks.values()) / len(self.code_blocks),
            "total_token_count": sum(block.token_count for block in self.code_blocks.values()),
        })

    async def analyze_blocks_in_parallel(self) -> List[str]:
        with ProcessPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(executor, self.analyze_block, block) 
                     for block in self.code_blocks.values()]
            return await asyncio.gather(*tasks)

    def analyze_block(self, block: CodeBlock) -> str:
        context = json.dumps(asdict(block))
        analysis = self.llm_api.analyze(
            context,
            "Describe the business processes implemented in this code block"
        )
        return f"Analysis of {block.id}:\n{analysis}"

    async def interactive_analysis(self):
        while True:
            query = input("Enter your question about the business processes (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            response = await self.llm_api.analyze(self.get_overall_summary(), query)
            print(f"Analysis: {response}")

    def visualize_graphs(self):
        plt.figure(figsize=(20, 10))
        
        plt.subplot(221)
        nx.draw(self.call_graph, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, arrows=True)
        plt.title("Call Graph")
        
        plt.subplot(222)
        nx.draw(self.data_flow_graph, with_labels=True, node_color='lightgreen', 
                node_size=500, font_size=8, arrows=True)
        plt.title("Data Flow Graph")
        
        plt.subplot(223)
        complexities = [block.complexity for block in self.code_blocks.values()]
        sns.histplot(complexities, kde=True)
        plt.title("Distribution of Code Complexity")
        plt.xlabel("Complexity")
        
        plt.subplot(224)
        token_counts = [block.token_count for block in self.code_blocks.values()]
        sns.scatterplot(x=complexities, y=token_counts)
        plt.title("Complexity vs Token Count")
        plt.xlabel("Complexity")
        plt.ylabel("Token Count")
        
        plt.tight_layout()
        plt.savefig('code_analysis_graphs.png')
        plt.close()

    async def load_domain_knowledge(self, file_path: str):
        with open(file_path, 'r') as f:
            self.domain_knowledge = json.load(f)

    async def analyze_with_domain_knowledge(self) -> str:
        analysis = await self.llm_api.analyze(
            json.dumps({
                "code_summary": self.get_overall_summary(),
                "domain_knowledge": self.domain_knowledge
            }),
            "Analyze the business processes in the context of the provided domain knowledge"
        )
        return analysis

    async def extract_business_processes(self) -> Dict[str, str]:
        processes = {}
        for block_id, block in self.code_blocks.items():
            business_logic = await self.llm_api.extract_business_logic(block.content)
            processes[block_id] = business_logic
        return processes

    def cluster_code_blocks(self, n_clusters=5):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([block.content for block in self.code_blocks.values()])
        
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(tfidf_matrix)
        
        clusters = defaultdict(list)
        for block_id, cluster in zip(self.code_blocks.keys(), kmeans.labels_):
            clusters[cluster].append(block_id)
        
        return clusters


    def generate_business_process_diagram(self):
        dot = Digraph(comment='Business Process Diagram')
        for process, description in self.business_processes.items():
            dot.node(process, description[:30] + '...')  # Truncate long descriptions
        
        for source, target, data in self.call_graph.edges(data=True):
            dot.edge(source, target)
        
        dot.render('business_process_diagram', format='png', cleanup=True)
        return 'business_process_diagram.png'

    async def analyze_code_evolution(self, git_repo_path: str) -> str:
        import git
        repo = git.Repo(git_repo_path)
        commits = list(repo.iter_commits('master', max_count=10))  # Analyze last 10 commits
        
        evolution_data = []
        for commit in commits:
            repo.git.checkout(commit.hexsha)
            await self.load_code(git_repo_path)
            summary = self.get_overall_summary()
            evolution_data.append({
                "commit": commit.hexsha,
                "date": commit.committed_datetime,
                "summary": summary
            })
        
        repo.git.checkout('master')  # Return to master branch
        
        evolution_analysis = await self.llm_api.analyze(
            json.dumps(evolution_data),
            "Analyze the evolution of the codebase based on these historical snapshots"
        )
        
        return evolution_analysis

    def generate_complexity_heatmap(self):
        complexities = {block.id: block.complexity for block in self.code_blocks.values()}
        df = pd.DataFrame.from_dict(complexities, orient='index', columns=['complexity'])
        df = df.sort_values('complexity', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.T, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Code Complexity Heatmap')
        plt.xlabel('Code Blocks')
        plt.ylabel('Complexity')
        plt.tight_layout()
        plt.savefig('complexity_heatmap.png')
        plt.close()
        return 'complexity_heatmap.png'

    async def identify_critical_paths(self):
        critical_paths = nx.all_simple_paths(self.call_graph, 
                                             source=list(self.call_graph.nodes())[0],
                                             target=list(self.call_graph.nodes())[-1])
        
        path_analyses = []
        for path in critical_paths:
            path_content = "\n".join([self.code_blocks[node].content for node in path])
            analysis = await self.llm_api.analyze(
                path_content,
                "Analyze this execution path and its business significance"
            )
            path_analyses.append({"path": path, "analysis": analysis})
        
        return path_analyses

    def calculate_code_churn(self, git_repo_path: str) -> Dict[str, int]:
        import git
        repo = git.Repo(git_repo_path)
        churn = {}
        for file_path in self.code_blocks.keys():
            try:
                blame = repo.git.blame('--line-porcelain', file_path)
                churn[file_path] = len(set(line.split()[0] for line in blame.splitlines() if line.startswith('author ')))
            except git.exc.GitCommandError:
                churn[file_path] = 0
        return churn

    async def analyze_code_quality(self):
        from pylint.lint import Run
        from pylint.reporters.text import TextReporter
        
        class CustomReporter(TextReporter):
            def __init__(self):
                super().__init__()
                self.messages = []

            def handle_message(self, msg):
                self.messages.append(msg)
        
        quality_reports = {}
        for block_id, block in self.code_blocks.items():
            reporter = CustomReporter()
            Run([block_id], reporter=reporter, exit=False)
            quality_reports[block_id] = reporter.messages
        
        quality_analysis = await self.llm_api.analyze(
            json.dumps(quality_reports),
            "Analyze the overall code quality based on these Pylint reports"
        )
        
        return quality_analysis

    def generate_word_cloud(self):
        from wordcloud import WordCloud
        
        all_text = " ".join([block.content for block in self.code_blocks.values()])
        stopwords = set(stopwords.words('english'))
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig('code_wordcloud.png')
        plt.close()
        return 'code_wordcloud.png'

async def main():
    llm_api = LLMApi()
    analyzer = BusinessProcessAnalyzer(llm_api)
    
    parser = argparse.ArgumentParser(description="Analyze source code for business processes")
    parser.add_argument("--path", required=True, help="Path to the source code directory")
    parser.add_argument("--domain-knowledge", help="Path to the domain knowledge JSON file")
    parser.add_argument("--git-repo", help="Path to the Git repository")
    args = parser.parse_args()

    try:
        # Load and analyze code
        await analyzer.load_code(args.path)
        
        # Perform business process analysis
        analysis_result = await analyzer.analyze_business_processes()
        print(analysis_result)
        
        # Generate visualizations
        analyzer.visualize_graphs()
        business_process_diagram = analyzer.generate_business_process_diagram()
        complexity_heatmap = analyzer.generate_complexity_heatmap()
        word_cloud = analyzer.generate_word_cloud()
        
        print(f"Business process diagram saved as {business_process_diagram}")
        print(f"Complexity heatmap saved as {complexity_heatmap}")
        print(f"Word cloud saved as {word_cloud}")
        
        # Cluster code blocks
        clusters = analyzer.cluster_code_blocks()
        print("Code block clusters:", json.dumps(clusters, indent=2))
        
        # Analyze with domain knowledge if provided
        if args.domain_knowledge:
            await analyzer.load_domain_knowledge(args.domain_knowledge)
            domain_analysis = await analyzer.analyze_with_domain_knowledge()
            print("Analysis with domain knowledge:", domain_analysis)
        
        # Analyze code evolution if Git repository is provided
        if args.git_repo:
            evolution_analysis = await analyzer.analyze_code_evolution(args.git_repo)
            print("Code evolution analysis:", evolution_analysis)
            
            code_churn = analyzer.calculate_code_churn(args.git_repo)
            print("Code churn analysis:", json.dumps(code_churn, indent=2))
        
        # Identify and analyze critical paths
        critical_paths = await analyzer.identify_critical_paths()
        print("Critical paths analysis:", json.dumps(critical_paths, indent=2))
        
        # Analyze code quality
        quality_analysis = await analyzer.analyze_code_quality()
        print("Code quality analysis:", quality_analysis)
        
        # Interactive analysis
        await analyzer.interactive_analysis()

    except Exception as e:
        logger.error(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())