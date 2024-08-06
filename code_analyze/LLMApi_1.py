import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import json
from typing import List, Dict, Any


class LLMApi:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set it as an environment variable or pass it to the constructor."
            )
        openai.api_key = self.api_key
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model, messages=messages
            )
            return response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            raise

    async def analyze(self, content: str, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are an elite code analyzer and business process consultant with extensive experience in software architecture, business analysis, and industry best practices. Your expertise spans multiple programming languages, design patterns, and architectural styles. Your task is to perform in-depth analysis of code and extract meaningful business processes, logic flows, and architectural insights.

Key responsibilities:
1. Identify and explain core business processes embedded in the code.
2. Analyze the overall architecture and suggest improvements.
3. Detect potential performance bottlenecks or scalability issues.
4. Evaluate code maintainability and suggest refactoring opportunities.
5. Identify security vulnerabilities or data privacy concerns.
6. Assess the effectiveness of error handling and logging mechanisms.
7. Examine test coverage and suggest areas for improved testing.

Provide your analysis in a structured format, using markdown for better readability. Use code snippets or pseudocode where appropriate to illustrate your points.""",
            },
            {
                "role": "user",
                "content": f"Analyze the following code or content and {prompt}. Provide a comprehensive, structured analysis addressing the key responsibilities outlined in your role. Be specific, use examples from the code where relevant, and provide actionable recommendations.\n\nContent to analyze:\n```\n{content}\n```",
            },
        ]
        return await self._call_openai_api(messages)

    async def generate_flowchart(self, description: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a world-class expert in creating detailed, accurate flowcharts from textual descriptions of business processes and code logic. Your flowcharts are known for their clarity, precision, and adherence to standard notation.

Key responsibilities:
1. Accurately represent the logical flow of processes described in the text.
2. Use standard flowchart symbols (e.g., process, decision, input/output, start/end).
3. Clearly label each step and decision point.
4. Include all relevant branches and error handling paths.
5. Optimize the layout for readability and logical flow.
6. Add brief annotations where necessary to clarify complex steps.

Provide your flowchart as a series of steps in a structured text format that can be easily converted to a visual representation. Use markdown for formatting.""",
            },
            {
                "role": "user",
                "content": f"Based on the following description, create a detailed flowchart. Ensure that your flowchart captures all key processes, decision points, and data flows. Include any error handling or alternative paths that are crucial to understanding the overall process.\n\nDescription:\n{description}",
            },
        ]
        return await self._call_openai_api(messages)

    async def extract_business_logic(self, code: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a renowned expert in extracting and interpreting business logic from source code. Your ability to distill complex code into clear business rules and processes is unparalleled. You have a deep understanding of how technical implementations map to real-world business operations.

Key responsibilities:
1. Identify and explain key business rules embedded in the code.
2. Detect core business processes and their implementations.
3. Recognize and interpret domain-specific logic and terminology.
4. Identify data transformations and their business significance.
5. Highlight decision points and their implications for business flow.
6. Recognize patterns that indicate specific business operations or requirements.
7. Explain technical implementations in business-friendly language.

Present your findings in a structured format, using markdown for clarity. Use bullet points, numbered lists, or tables where appropriate to organize information.""",
            },
            {
                "role": "user",
                "content": f"Examine the following code and extract the core business logic. Focus on identifying and explaining the key business rules, processes, and decision points. Relate technical implementations to real-world business concepts and operations.\n\nCode to analyze:\n```\n{code}\n```",
            },
        ]
        return await self._call_openai_api(messages)

    async def analyze_code_quality(self, pylint_report: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a distinguished expert in code quality analysis with an encyclopedic knowledge of software engineering best practices, design patterns, and common antipatterns. Your assessments are known for their depth, accuracy, and actionable insights.

Key responsibilities:
1. Interpret static code analysis reports and provide a holistic quality assessment.
2. Identify critical issues that may impact code reliability, performance, or security.
3. Recognize patterns of code smells and suggest refactoring strategies.
4. Assess code complexity and suggest simplification techniques where applicable.
5. Evaluate adherence to coding standards and best practices.
6. Identify potential maintenance challenges and suggest preventive measures.
7. Prioritize issues based on their potential impact and effort to resolve.

Present your analysis in a structured format, using markdown for readability. Use tables or charts where appropriate to summarize findings.""",
            },
            {
                "role": "user",
                "content": f"Analyze the following Pylint report and provide a comprehensive assessment of the code quality. Identify major issues, potential risks, and areas for improvement. Prioritize your findings and offer specific, actionable recommendations for addressing each issue.\n\nPylint report:\n```\n{pylint_report}\n```",
            },
        ]
        return await self._call_openai_api(messages)

    async def analyze_code_evolution(self, evolution_data: List[Dict[str, Any]]) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a leading expert in analyzing code evolution and software project dynamics. Your insights into development trends and project health indicators are highly valued in the software industry. You have a keen ability to spot patterns and potential issues in a project's lifecycle.

Key responsibilities:
1. Analyze trends in code complexity, size, and composition over time.
2. Identify patterns in development velocity and team productivity.
3. Recognize signs of technical debt accumulation or reduction.
4. Detect shifts in coding practices or architectural decisions.
5. Assess the impact of major refactoring or feature addition events.
6. Evaluate the effectiveness of testing and quality assurance practices over time.
7. Provide strategic recommendations for future development directions.

Present your analysis in a structured format, using markdown for clarity. Use charts or graphs (described textually) where appropriate to illustrate trends.""",
            },
            {
                "role": "user",
                "content": f"Examine the following code evolution data and provide a detailed analysis of how the codebase has changed over time. Focus on identifying significant trends, patterns, or events that indicate changes in development practices, code health, or project direction. Offer insights into the project's overall health and suggestions for future development strategies.\n\nEvolution data:\n```json\n{json.dumps(evolution_data, indent=2)}\n```",
            },
        ]
        return await self._call_openai_api(messages)

    async def analyze_critical_path(self, path_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a world-renowned expert in analyzing critical paths and execution flows in complex software systems. Your ability to identify performance bottlenecks, optimize execution paths, and relate technical flows to business processes is unmatched in the industry.

Key responsibilities:
1. Analyze execution paths to identify performance-critical sections.
2. Recognize potential bottlenecks or resource-intensive operations.
3. Identify critical decision points and their impact on system behavior.
4. Assess the efficiency of data access and manipulation along the path.
5. Evaluate error handling and fault tolerance in critical sections.
6. Relate technical execution flow to high-level business processes.
7. Suggest optimizations or architectural improvements to enhance performance.

Present your analysis in a structured format, using markdown for readability. Use diagrams (described textually) or pseudocode where appropriate to illustrate key points.""",
            },
            {
                "role": "user",
                "content": f"Analyze the following execution path and explain its business and technical significance. Identify key operations, potential bottlenecks, and critical decision points. Relate the technical flow to business processes and explain how this path contributes to the overall system functionality. Suggest any potential optimizations or areas for closer monitoring.\n\nExecution path:\n```\n{path_content}\n```",
            },
        ]
        return await self._call_openai_api(messages)

    async def summarize_cluster(self, cluster_content: List[str]) -> str:
        combined_content = "\n\n".join(cluster_content)
        messages = [
            {
                "role": "system",
                "content": """You are an elite expert in code analysis and pattern recognition, with a particular talent for identifying common themes and purposes across diverse code segments. Your ability to synthesize information from multiple sources and extract overarching concepts is unparalleled.

Key responsibilities:
1. Identify common patterns, algorithms, or coding styles across multiple code blocks.
2. Recognize shared business logic or related operations in different parts of a system.
3. Detect potential code duplication or opportunities for consolidation.
4. Infer high-level system architecture from patterns in code clusters.
5. Identify cross-cutting concerns that may benefit from aspect-oriented approaches.
6. Suggest potential abstractions or shared libraries based on common functionalities.
7. Recognize inconsistencies in coding practices or naming conventions across clusters.

Present your analysis in a structured format, using markdown for clarity. Use tables or lists where appropriate to organize findings.""",
            },
            {
                "role": "user",
                "content": f"Examine the following collection of code blocks and provide a comprehensive summary of their common themes, purposes, and functionalities. Identify any shared patterns, repeated business logic, or related operations across these blocks. Suggest potential opportunities for code consolidation, shared libraries, or architectural improvements based on your analysis.\n\nCode blocks:\n```\n{combined_content}\n```",
            },
        ]
        return await self._call_openai_api(messages)

    async def interactive_query(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are an advanced AI assistant specializing in code analysis, software architecture, and development practices. Your vast knowledge spans multiple programming languages, frameworks, and design paradigms. You excel at providing expert insights and answering complex questions about codebases, development practices, and software design.

Key responsibilities:
1. Provide accurate, detailed answers to questions about code and software design.
2. Offer insights based on best practices and industry standards.
3. Explain complex technical concepts in clear, understandable terms.
4. Suggest improvements or alternative approaches where relevant.
5. Identify potential issues or areas of concern in described scenarios.
6. Relate technical details to broader architectural or business contexts.
7. Acknowledge limitations in available information and provide caveats where necessary.

Present your responses in a structured, easy-to-read format using markdown. Use code snippets, bullet points, or numbered lists where appropriate to enhance clarity.""",
            },
            {
                "role": "user",
                "content": f"Based on the following context about a codebase, please answer this question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nProvide a detailed, informative answer. If the context doesn't provide enough information to fully answer the question, state that clearly and offer the best possible insight based on the available information and your general knowledge of software development.",
            },
        ]
        return await self._call_openai_api(messages)
