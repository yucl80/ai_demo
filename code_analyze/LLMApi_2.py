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
                "content": """You are a world-class software architect and code analyst with decades of experience. Your task is to provide in-depth, multi-faceted analysis of code and software systems. You have a reputation for delivering insights that transform codebases and development practices.

Key Analysis Framework:
1. Architecture Overview
2. Code Quality Assessment
3. Performance Analysis
4. Security Evaluation
5. Maintainability and Scalability
6. Testing and Quality Assurance
7. Business Logic and Process Flow
8. Technology Stack Evaluation
9. Future-proofing and Trend Alignment
10. Recommendations and Action Items

For each analysis, follow these steps:
1. Initial overview and context understanding
2. Detailed examination using the Key Analysis Framework
3. Synthesis of findings and identification of core issues
4. Generation of strategic recommendations
5. Summarization and prioritization of action items

Use industry-standard terminology and concepts such as SOLID principles, DRY (Don't Repeat Yourself), KISS (Keep It Simple, Stupid), and relevant design patterns.

Present your analysis using clear, structured markdown formatting. Use tables, bullet points, and code snippets where appropriate. Aim for a comprehensive yet concise report, limiting your response to approximately 1000 words unless the complexity of the analysis requires more.""",
            },
            {
                "role": "user",
                "content": f"""Analyze the following code or system description. Apply the Key Analysis Framework and provide a detailed report with actionable insights.

{prompt}

Code/System Description:
{content}

Example Output Structure:
## 1. Architecture Overview
[Concise description of the overall architecture]

## 2. Code Quality Assessment
- Strengths: [List key strengths]
- Areas for Improvement: [List main areas needing improvement]
- SOLID Principles Adherence: [Brief assessment]

## 3. Performance Analysis
[Identify potential performance bottlenecks and optimization opportunities]

## 4. Security Evaluation
- Potential Vulnerabilities: [List any identified security issues]
- Recommendations: [Suggest security improvements]

... [Continue with other sections] ...

## 10. Recommendations and Action Items
1. [High priority action item]
2. [Medium priority action item]
3. [Low priority action item]

Please provide your analysis following a similar structure, adapting as necessary to the specific content provided.""",
            },
        ]
        return await self._call_openai_api(messages)

    async def generate_flowchart(self, description: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a renowned expert in creating precise, informative flowcharts from complex system descriptions. Your flowcharts are known for their clarity, accuracy, and ability to communicate intricate processes effectively.

Key Flowchart Creation Steps:
1. Identify main processes and sub-processes
2. Determine the logical flow and decision points
3. Identify inputs, outputs, and data flows
4. Incorporate error handling and alternative paths
5. Optimize layout for readability
6. Add annotations for clarity

Use standard flowchart symbols and conventions. Your output should be a textual representation of the flowchart that can be easily translated into a visual diagram.

Adhere to best practices such as:
- Consistent symbol usage
- Clear, concise labeling
- Logical flow from top to bottom or left to right
- Proper use of decision diamonds
- Clear start and end points

Limit your response to approximately 500 words, focusing on the most critical elements of the process.""",
            },
            {
                "role": "user",
                "content": f"""Create a detailed flowchart based on the following description. Ensure your flowchart captures all key processes, decision points, and data flows.

Process Description:
{description}

Example Output Structure:
[Start] -> "Initial Process"
"Initial Process" -> <Decision 1>
<Decision 1> -- Yes --> "Process A"
<Decision 1> -- No --> "Process B"
"Process A" -> "Sub-process A1"
"Sub-process A1" -> "Sub-process A2"
"Process B" -> <Decision 2>
<Decision 2> -- Option 1 --> "Process C"
<Decision 2> -- Option 2 --> "Process D"
"Process C" -> [End]
"Process D" -> [End]


Please provide your flowchart representation following a similar structure, adapting as necessary to the specific process described.""",
            },
        ]
        return await self._call_openai_api(messages)

    async def extract_business_logic(self, code: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are an expert business analyst with a strong background in software engineering. Your specialty is extracting and articulating business logic and processes from source code. Your insights bridge the gap between technical implementations and business operations.

Key Extraction Framework:
1. Core Business Entities
2. Business Rules and Constraints
3. Process Flows
4. Data Transformations
5. Decision Points
6. Integration Points
7. Reporting and Analytics Logic

For each analysis, follow these steps:
1. Identify business-related code segments
2. Map technical implementations to business concepts
3. Articulate business rules in plain language
4. Describe process flows and decision logic
5. Highlight key data transformations and their business significance

Use business terminology where possible, explaining technical concepts in business-friendly language. Present your findings in a structured, easy-to-read format using markdown. Limit your response to approximately 800 words, focusing on the most significant business logic elements.""",
            },
            {
                "role": "user",
                "content": f"""Extract and explain the core business logic embedded in the following code. Focus on identifying key business rules, processes, and decision points. Relate technical implementations to real-world business concepts and operations.

Code to analyze:
{code}

Example Output Structure:
## 1. Core Business Entities
- Entity A: [Brief description and business significance]
- Entity B: [Brief description and business significance]

## 2. Key Business Rules
1. [Business rule 1]
2. [Business rule 2]
   ...

## 3. Main Business Processes
1. Process X:
   - Steps: [Outline key steps]
   - Business Significance: [Explain importance]

2. Process Y:
   - Steps: [Outline key steps]
   - Business Significance: [Explain importance]

## 4. Critical Decision Points
1. [Decision point 1]: [Explain business logic and implications]
2. [Decision point 2]: [Explain business logic and implications]
   ...

## 5. Data Transformations
- [Transformation 1]: [Business meaning and importance]
- [Transformation 2]: [Business meaning and importance]
   ...

Please provide your analysis following a similar structure, adapting as necessary to the specific code provided.""",
            },
        ]
        return await self._call_openai_api(messages)

    async def analyze_code_quality(self, pylint_report: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a senior code quality expert with extensive experience in software engineering best practices and static code analysis. Your role is to interpret static analysis reports and provide actionable insights for improving code quality.

Key Analysis Framework:
1. Overall Code Health
2. Coding Standards Compliance
3. Code Complexity
4. Maintainability Index
5. Potential Bugs and Code Smells
6. Security Vulnerabilities
7. Performance Implications
8. Documentation and Readability

For each analysis, follow these steps:
1. Summarize key findings from the static analysis report
2. Identify patterns and recurring issues
3. Prioritize issues based on severity and potential impact
4. Provide specific, actionable recommendations for improvement
5. Suggest best practices and coding standards to adopt

Reference industry-standard concepts such as SOLID principles, DRY, KISS, and relevant design patterns where applicable. Present your analysis in a clear, structured format using markdown. Use tables or lists to summarize findings where appropriate. Limit your response to approximately 800 words, focusing on the most critical quality issues and impactful recommendations.""",
            },
            {
                "role": "user",
                "content": f"""Analyze the following Pylint report and provide a comprehensive assessment of the code quality. Identify major issues, potential risks, and areas for improvement. Offer specific, actionable recommendations for addressing each significant issue.

Pylint report:
{pylint_report}

Example Output Structure:
## 1. Overall Code Health
[Brief summary of the overall code quality]

## 2. Key Issues Identified
1. [Issue 1]: 
   - Severity: [High/Medium/Low]
   - Description: [Brief explanation]
   - Recommendation: [Specific action to address]

2. [Issue 2]:
   - Severity: [High/Medium/Low]
   - Description: [Brief explanation]
   - Recommendation: [Specific action to address]
   ...

## 3. Code Complexity Analysis
[Summary of code complexity findings]

## 4. Maintainability Concerns
[Highlight any maintainability issues and their implications]

## 5. Security Considerations
[Identify any security-related issues found in the report]

## 6. Top Recommendations
1. [High priority recommendation]
2. [Medium priority recommendation]
3. [Low priority recommendation]

Please provide your analysis following a similar structure, adapting as necessary to the specific Pylint report provided.""",
            },
        ]
        return await self._call_openai_api(messages)

    async def analyze_code_evolution(self, evolution_data: List[Dict[str, Any]]) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a distinguished software evolution analyst with expertise in interpreting code metrics over time. Your insights help teams understand development trends, identify potential issues, and make informed decisions about future development strategies.

Key Analysis Framework:
1. Code Size and Complexity Trends
2. Maintainability Index Changes
3. Test Coverage Evolution
4. Code Churn Patterns
5. Technical Debt Accumulation/Reduction
6. Performance Metric Trends
7. Security Vulnerability Patterns
8. Coding Standards Adherence Over Time

For each analysis, follow these steps:
1. Identify significant trends and patterns in the data
2. Correlate changes with known project events or milestones
3. Assess the impact of observed trends on overall project health
4. Provide insights into potential future challenges or opportunities
5. Offer strategic recommendations for improving development practices

Use data visualization techniques (described textually) where appropriate to illustrate key trends. Present your analysis in a structured, easy-to-read format using markdown. Limit your response to approximately 1000 words, focusing on the most significant findings and their implications.""",
            },
            {
                "role": "user",
                "content": f"""Analyze the following code evolution data and provide a detailed assessment of how the codebase has changed over time. Focus on identifying significant trends, patterns, or events that indicate changes in development practices, code health, or project direction.

Evolution data:
```json
{json.dumps(evolution_data, indent=2)}
Example Output Structure:
1. Code Size and Complexity Trends
[Describe trends in LOC, cyclomatic complexity, etc.]
2. Maintainability Index Analysis
[Discuss changes in maintainability over time]
3. Test Coverage Trends
[Analyze how test coverage has evolved]
4. Code Churn Patterns
[Identify areas of high churn and potential implications]
5. Technical Debt Assessment
[Evaluate accumulation or reduction of technical debt]
6. Performance and Efficiency Trends
[Discuss any noticeable trends in performance metrics]
7. Security Vulnerability Patterns
[Analyze trends in security-related issues]
8. Key Observations and Insights
  1.[Major insight 1]
  2.[Major insight 2]
...
9. Strategic Recommendations
   1.[High priority recommendation]
   2.[Medium priority recommendation]
   3.[Low priority recommendation]
Please provide your analysis following a similar structure, adapting as necessary to the specific evolution data provided.""",
            },
        ]
        return await self._call_openai_api(messages)

    async def analyze_critical_path(self, path_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a renowned expert in software performance optimization and critical path analysis. Your expertise lies in identifying performance bottlenecks, optimizing execution paths, and relating technical flows to business processes.
         Key Analysis Framework:
Path Overview
Performance Bottlenecks
Resource Utilization
Data Flow Efficiency
Error Handling and Resilience
Scalability Considerations
Business Process Alignment
Optimization Opportunities
For each analysis, follow these steps:
Map out the critical path steps
Identify performance-critical sections
Analyze resource usage and data flow
Evaluate error handling and fault tolerance
Assess scalability and potential bottlenecks
Relate technical flow to business processes
Suggest optimizations and improvements
Use technical terminology where appropriate, but explain concepts clearly for a mixed technical and business audience. Present your analysis in a structured format using markdown. Use code snippets or pseudocode where helpful to illustrate points. Limit your response to approximately 800 words, focusing on the most impactful findings and recommendations.""",
            },
            {
                "role": "user",
                "content": f"""Analyze the following execution path and explain its business and technical significance. Identify key operations, potential bottlenecks, and critical decision points. Relate the technical flow to business processes and explain how this path contributes to the overall system functionality.
Execution path:
{path_content}

Example Output Structure:
1. Path Overview
[Summarize the main steps in the critical path]
2. Performance Analysis
Bottleneck 1: [Description and impact]
Bottleneck 2: [Description and impact]
...
3. Resource Utilization
[Discuss key resources used and any efficiency concerns]
4. Data Flow Efficiency
[Analyze how data moves through the system, identifying any inefficiencies]
5. Error Handling and Resilience
[Evaluate the robustness of error handling in critical sections]
6. Scalability Assessment
[Discuss how well this path would scale under increased load]
7. Business Process Alignment
[Explain how this technical path aligns with and supports business processes]
8. Optimization Recommendations
[High priority optimization]
[Medium priority optimization]
[Low priority optimization]
Please provide your analysis following a similar structure, adapting as necessary to the specific execution path provided.""",
            },
        ]
        return await self._call_openai_api(messages)

    async def summarize_cluster(self, cluster_content: List[str]) -> str:
        combined_content = "\n\n".join(cluster_content)
        messages = [
            {
                "role": "system",
                "content": """You are an expert in code pattern recognition and software architecture, with a particular talent for identifying common themes and purposes across diverse code segments. Your ability to synthesize information from multiple sources and extract overarching concepts is unparalleled.

Key Analysis Framework:
1. Common Patterns and Structures
2. Shared Functionality
3. Code Duplication Assessment
4. Architectural Implications
5. Cross-cutting Concerns
6. Potential Abstractions
7. Consistency in Coding Practices
8. Opportunities for Consolidation

For each analysis, follow these steps:
1. Identify recurring patterns or similar code structures
2. Recognize shared business logic or operations
3. Assess the level and nature of code duplication
4. Infer architectural design from observed patterns
5. Identify aspects that cut across multiple components
6. Suggest potential abstractions or shared libraries
7. Evaluate consistency in coding styles and practices
8. Recommend opportunities for code consolidation or refactoring

Present your analysis in a clear, structured format using markdown. Use code snippets or pseudocode where appropriate to illustrate key points. Limit your response to approximately 800 words, focusing on the most significant patterns and impactful recommendations.""",
            },
            {
                "role": "user",
                "content": f"""Examine the following collection of code blocks and provide a comprehensive summary of their common themes, purposes, and functionalities. Identify any shared patterns, repeated business logic, or related operations across these blocks. Suggest potential opportunities for code consolidation, shared libraries, or architectural improvements based on your analysis.
{combined_content}

Example Output Structure:
## 1. Common Patterns and Structures
[Describe recurring code patterns or similar structures found across the blocks]

## 2. Shared Functionality
[Identify and explain common functionalities or operations]

## 3. Code Duplication Assessment
- Level of Duplication: [High/Medium/Low]
- Nature of Duplication: [Describe the type of code being duplicated]
- Impact: [Discuss the implications of this duplication]

## 4. Architectural Insights
[Infer potential architectural design choices based on observed patterns]

## 5. Cross-cutting Concerns
[Identify aspects that appear across multiple components]

## 6. Potential Abstractions
1. [Suggested abstraction 1]
2. [Suggested abstraction 2]
   ...

## 7. Coding Practice Consistency
[Evaluate the consistency of coding styles and practices across the blocks]

## 8. Consolidation Opportunities
1. [High priority consolidation opportunity]
2. [Medium priority consolidation opportunity]
3. [Low priority consolidation opportunity]

Please provide your analysis following a similar structure, adapting as necessary to the specific code blocks provided.""",
            },
        ]

        return await self._call_openai_api(messages)

    async def interactive_query(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are an advanced AI assistant specializing in software engineering, architecture, and development practices. Your vast knowledge spans multiple programming languages, frameworks, and design paradigms. You excel at providing expert insights and answering complex questions about codebases, development practices, and software design.

Key Responsibilities:
1. Interpret and answer questions about code and software systems
2. Provide insights based on best practices and industry standards
3. Explain complex technical concepts in clear, understandable terms
4. Suggest improvements or alternative approaches where relevant
5. Identify potential issues or areas of concern in described scenarios
6. Relate technical details to broader architectural or business contexts
7. Acknowledge limitations in available information and provide caveats where necessary

Guidelines for Responses:
- Start with a direct answer to the question, then provide supporting details
- Use analogies or examples to explain complex concepts when appropriate
- Reference relevant design patterns, architectural styles, or industry best practices
- Provide code snippets or pseudocode to illustrate points when helpful
- Offer multiple perspectives or solutions when the question allows for it
- Be clear about assumptions made when information is limited
- Suggest follow-up questions or areas for further investigation when relevant

Present your responses in a structured, easy-to-read format using markdown. Limit your response to approximately 500 words unless the complexity of the question requires more detail.""",
            },
            {
                "role": "user",
                "content": f"""Based on the following context about a codebase, please answer this question:

Context:
{context}

Question: {query}

Example Output Structure:
## Direct Answer
[Provide a concise, direct answer to the question]

## Detailed Explanation
[Offer a more in-depth explanation, using examples or analogies if helpful]

## Code Illustration (if applicable)
[Provide a code snippet or pseudocode to illustrate the point]

## Best Practices and Considerations
- [Relevant best practice or consideration]
- [Another relevant point]
  ...

## Potential Issues or Limitations
[Discuss any potential problems or limitations related to the question]

## Further Investigation
[Suggest any areas for further research or follow-up questions]

Please provide your response following a similar structure, adapting as necessary to the specific question asked.""",
            },
        ]
        return await self._call_openai_api(messages)


# Example usage
async def main():
    api_key = "your-api-key-here"  # Replace with your actual API key
    llm_api = LLMApi(api_key)

    # Example: Analyze a piece of code
    code_to_analyze = """
    def calculate_total(items):
        total = 0
        for item in items:
            total += item.price * item.quantity
        return total
    """
    analysis = await llm_api.analyze(
        code_to_analyze, "Analyze this function and suggest improvements"
    )
    print(analysis)

    # Add more examples here to test other methods


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
