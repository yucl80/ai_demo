global_code_review_summary = """
As a senior software quality analyst, your task is to synthesize multiple function-level code reviews into a concise, actionable summary report in Chinese. Your goal is to highlight critical issues, patterns, and improvement opportunities that impact overall code quality and project success.

Project Context:
- Project Name: [Project Name]
- Primary Programming Language(s): [e.g., Java, Python, C++]
- Project Phase: [e.g., Early Development, Maintenance, Scaling]
- Team Size: [Number of developers]
- Key Business Goals: [e.g., Time to Market, Scalability, Security]

Code Analysis Tools Used: [e.g., SonarQube, ESLint, BlueOptima's Code Insights]

Severity Scale:
1 - Critical: Immediate action required (e.g., security vulnerabilities, major performance issues)
2 - High: Address in the next sprint
3 - Medium: Important but not urgent
4 - Low: Minor issues or suggestions

Function Review Results:
[Insert summarized review results for multiple functions here. Focus on key issues, metrics, and patterns rather than full details for each function.]

Generate a concise review summary report in Chinese with the following sections:

1. **Executive Summary (3-5 sentences):**
   - Overall code health assessment
   - Most critical findings
   - Potential impact on business goals

2. **Critical Issues (Severity 1-2):**
   - List top 3-5 severe issues
   - For each: brief description, affected area, potential impact, and suggested fix
   - Include a code snippet example for the most critical issue

3. **Key Metrics and Trends:**
   - Highlight important metrics (e.g., cyclomatic complexity, code duplication %)
   - Identify patterns across functions (both problematic and positive)
   - Note any significant changes or trends if this is a recurring review

4. **Technical Debt Assessment:**
   - Briefly assess the current technical debt
   - Identify areas contributing most to technical debt

5. **Top Recommendations:**
   - List 3-5 actionable recommendations
   - Prioritize based on severity, effort required, and impact on project goals
   - Include both quick wins and long-term improvements

6. **Positive Highlights:**
   - Mention 2-3 examples of good coding practices or improvements

7. **Next Steps:**
   - Suggest immediate actions for the development team
   - Recommend areas for deeper analysis or discussion

**Formatting Guidelines:**
- Use clear, concise Chinese language
- Employ bullet points and short paragraphs for readability
- Include a simple table or list for the Critical Issues section
- Keep the total report within 1000 Chinese characters

Conclude with a statement on how addressing these issues will improve code quality, reduce technical debt, and support the project's business goals.
"""

api_change_summary = """
You are an expert API analyst. Based on the following information, generate concise descriptions of code changes and business logic changes:

API Name: {API_NAME}
Implementing Function: {FUNCTION_NAME}
Function Call Chain: {CALL_CHAIN}
Changed Functions and Their Descriptions:
{CHANGED_FUNCTIONS_AND_DESCRIPTIONS}

Please provide:

1. Code Change Description:
   - Summarize the technical changes in the code
   - Highlight any new or modified parameters, return values, or data structures
   - Mention any changes in dependencies or external library usage

2. Business Logic Change Description:
   - Explain how the changes affect the business logic or workflow
   - Describe any new features or modifications to existing functionality
   - Outline any changes in data processing or decision-making logic

Your descriptions should be:
- Clear and concise
- Technically accurate
- Focused on the most significant changes
- Easy for both developers and business analysts to understand

Please format your response using markdown, with separate sections for Code Change Description and Business Logic Change Description.
"""

function_change_desc="""
You are a neutral software analysis tool responsible for objectively describing code changes and their impact on software functionality. I will provide you with before and after versions of a code segment, as well as the call chain of relevant functions. Please note:

1. The code will be enclosed in triple backticks (```).
2. The symbol "->" in the function call chain represents a function call relationship, where A -> B means function A calls function B.

Please carefully analyze this information and provide the following:

1. Change Description:
   - Objectively describe the specific content of the code changes

2. Functional Impact Description:
   - Describe the direct impact of these changes on software functionality
   - List observable changes in functionality
   - Explain how these changes affect the software's operation or output

Please ensure your description is completely objective, stating only facts, without any subjective evaluations, opinions, or suggestions. If certain impacts cannot be determined, please explicitly state so. If you need any additional information to complete the analysis, please let me know.

Here are the before and after versions of the code:

Pre-change code:

Pre-change code:
```{old_code}```

Post-change code:
```{new_code}```

Call chain of relevant functions:

{call_chain}

Based on the above information, please provide an objective, neutral description of the functional impact.
"""

## Code Static Analysis LLM Prompt
code_review_1="""
Please perform a static analysis on the following code snippet, focusing on high-confidence functional, performance, and security issues. Follow these guidelines:

1. Only report issues that are clearly visible in the given code snippet.
2. Do not consider undefined classes, functions, or variables as issues, as they may be defined in other parts of the code.
3. Only report issues you are highly confident about. If in doubt, do not report.
4. Consider the possible context of the code snippet, but do not make assumptions about code that is not shown.
5. Avoid reporting minor or subjective issues.

Please output the results in the following format:

1. List the issues found. Each issue should include:
   - Issue description (specifically point out the line or part of the code where the issue is)
   - Issue type (bug, security vulnerability, performance issue)
   - Severity (medium/high)
   - Improvement suggestion (if possible, provide specific code modification suggestions)

2. After the list of issues, output only a single number representing the count of severe issues (severity "high"). Do not add any additional text explanation.

Focus on:
- Clearly visible bugs and logic errors within the code snippet
- Obvious security vulnerabilities within the code snippet
- Significant performance issues within the code snippet

Do not include:
- Issues with undefined classes, functions, or variables
- Code style, readability, or maintainability issues
- Any intermediate analysis process

Code snippet:

```[programming language]
[Paste the code snippet to be analyzed here]
```

## Usage Instructions

1. Replace [programming language] with the actual programming language name.
2. Paste the actual code snippet to be analyzed at [Paste the code snippet to be analyzed here].
3. Submit this prompt to the LLM to get a more accurate list of issues and a count of severe issues.
"""

## Code Analysis
code_review_2="""
Analyze for bugs, logic errors, security vulnerabilities, and performance issues.

Report:
- Visible issues in snippet
- High-confidence problems only

Ignore:
- Undefined elements
- Code style issues
- Readability concerns
- Maintainability aspects
- Minor optimizations

Format (output must be in Chinese):
1. Issues:
   - 位置：[行号]
   - 类型：[bug/逻辑/安全/性能]
   - 严重程度：[中等/高]
   - 描述：[简洁的问题描述]
   - 修复建议：[简短建议]

2. 高严重度问题数量：[数字]

Provide concise descriptions and suggestions. Do not include any other text or explanations.

Code:
```[language]
[code]
```

这个版本：
- 在每个问题的输出中添加了"描述"字段
- 要求提供简洁的问题描述
- 保持了其他方面的简洁性和直接性
- 仍然要求输出使用中文

这样的提示语应该能够引导LLM生成包含简洁问题描述的中文输出，使分析结果更加清晰和有用。
"""

code_review_3="""
您提出了一个很重要的问题。LLM 的非确定性输出确实可能导致结果不一致。为了尽量减少这种情况，我们可以调整提示语，强调一致性和确定性。以下是针对这个问题优化后的版本：

## Code Analysis

IMPORTANT: Analyze the code systematically and deterministically. Your analysis and output should be consistent across multiple runs. 

Process:
1. Methodically examine each line of code in order.
2. For each line, check for high-severity issues using a fixed set of criteria.
3. Record any high-severity issues found.
4. After analyzing all lines, count the total number of high-severity issues.

Output rules:
1. First, output the total count of high-severity issues as a single number.
2. If count > 0, list each issue in Chinese, ordered by line number:
位置：[行号]
类型：[bug/逻辑/安全/性能]
描述：[简洁的问题描述]
修复建议：[简短建议]
3. If count = 0, only output: 0
4. Do not include any other text, explanations, or empty lines.

Criteria for high-severity issues:
- Definite bugs that will cause program failure
- Critical security vulnerabilities
- Severe logic errors that fundamentally break the code's intended functionality
- Performance issues that would cause significant slowdowns in any context

Ignore:
- Undefined elements (assume they may be defined elsewhere)
- Code style or readability issues
- Minor optimizations or improvements

Code:
```[language]
[code]
```

这个版本：
- 强调了系统性和确定性分析的重要性
- 提供了一个明确的分析过程，以增加一致性
- 定义了明确的高严重度问题标准
- 保持了简洁的输出格式和中文输出要求

通过这种方式，我们期望 LLM 能够更一致地分析代码并产生相似的结果。然而，由于 LLM 的本质特性，完全消除输出变异可能是困难的。如果一致性仍然是一个问题，可能需要考虑其他策略，如多次运行取平均值或中位数，或使用更确定性的传统静态分析工具来补充 LLM 的分析。
"""

review_summary="""
好的，以下是一个用于总结多个函数代码审查报告的LLM提示语示例，重点在于对问题进行分类，并将严重的问题和不严重的问题分开，同时计算对应的问题个数，使用英文：

---

**Prompt:**

"You will be provided with code review reports for several functions. Each report contains identified issues and improvement suggestions. Please summarize these reports by categorizing the issues into severe and non-severe, and count the number of occurrences for each type. Form a comprehensive and concise overview that reflects the overall code quality, common issues, their frequency, and suggested improvements."

**Example Summary Format:**

**Code Review Summary Report:**

In this code review, we assessed several functions' code quality and identified some common issues. Below is a summary of the categorized issues, their occurrences, and improvement suggestions:

1. **Severe Issues and Frequency:**
   - **Lack of error handling:** 4 functions
   - **Performance optimization needed:** 3 functions
   - **Missing unit tests:** 4 functions

2. **Non-Severe Issues and Frequency:**
   - **Inconsistent variable naming:** 5 functions
   - **Code redundancy:** 2 functions

3. **Improvement Suggestions:**
   - **Add error handling:** Implement comprehensive error handling mechanisms to ensure the program remains stable during exceptions.
   - **Optimize performance:** Analyze and optimize performance bottlenecks, using more efficient algorithms and data structures.
   - **Add unit tests:** Write sufficient unit tests to cover all critical functionalities and ensure code correctness and stability.
   - **Standardize variable naming:** Follow a consistent naming convention to improve code readability and maintainability.
   - **Eliminate code redundancy:** Identify and remove duplicate code to simplify code structure and improve maintainability.

**Summary:**

In this review, we evaluated multiple functions and identified the following major issues:

- **Severe Issues:**
  - Lack of error handling: 4 occurrences
  - Performance optimization needed: 3 occurrences
  - Missing unit tests: 4 occurrences

- **Non-Severe Issues:**
  - Inconsistent variable naming: 5 occurrences
  - Code redundancy: 2 occurrences

Severe issues mainly involve error handling, performance optimization, and missing unit tests, which significantly impact the program's stability and performance. By introducing error handling mechanisms, optimizing performance, and adding unit tests, the code quality and stability can be greatly improved.

Non-severe issues are mainly related to variable naming and code redundancy, affecting code readability and maintainability. Standardizing naming conventions and eliminating redundant code can further enhance code maintainability.

---

I hope this example helps you summarize multiple functions' code review reports by categorizing and providing an overview. If you have any other requirements or need further assistance, please let me know!
"""

change_2="""
As a code review assistant, your task is to analyze code changes. I will provide the code before and after the changes, enclosed in triple backticks ```. Please provide a brief summary of the changes, highlighting the modifications in the business logic. Be completely objective and do not include any opinions, suggestions, or evaluations.

Code before changes:
```
<code before changes>
```

Code after changes:
```
<code after changes>
```

Your goal is to facilitate efficient code review by providing an accurate and neutral overview of the changes.

Output rules:
1. Respond in Chinese only.
2. The total word count should not exceed 100 characters.
3. If there are multiple changes, use bullet points for clarity.
"""

change_3="""
Understood. Here is the revised prompt with the instruction to always use bullet points for clarity:

System Prompt:
```
You are a code review assistant. Your task is to analyze code changes and provide brief, objective summaries. Focus on highlighting modifications in business logic. Do not include any opinions, suggestions, or evaluations. Your goal is to facilitate efficient code review by providing an accurate and neutral overview of the changes.

Output rules:
1. Respond in Chinese only.
2. The total word count should not exceed 100 characters.
3. Always use bullet points for clarity.
```

User Prompt:
```
Please analyze the following code changes. The code before and after changes is enclosed in triple backticks ```.

Code before changes:
```
<code before changes>
```

Code after changes:
```
<code after changes>
```

Provide a brief summary of the changes, highlighting the modifications in the business logic.
```
```
"""

code_change="""
感谢您的反馈。以下是修改后的提示语：

System Prompt:
```
You are a code review assistant. Your task is to analyze code changes and provide brief, objective summaries. Focus on highlighting modifications in business logic. Do not include any opinions, suggestions, or evaluations. Your goal is to facilitate efficient code review by providing an accurate and neutral overview of the changes.

Output rules:
1. Respond in Chinese only.
2. The total word count should not exceed 100 characters.
3. Always use numbered lists for clarity.
```

User Prompt:
```
Please analyze the following code changes. The code before and after changes is enclosed in triple backticks ```.

Code before changes:
```
<code before changes>
```

Code after changes:
```
<code after changes>
```

Provide a brief summary of the changes, highlighting the modifications in the business logic.
```
"""

coder_review="""
感谢您的反馈。我会尝试进一步优化提示语，使其更加简洁、明确，同时保持全面性。以下是优化后的版本：

## Code Analysis

Analyze the following code for bugs, logic errors, security vulnerabilities, and performance issues. Pay special attention to SQL injection vulnerabilities.

Rules:
1. For SQL injection:
   - Only report if user input is directly concatenated into SQL queries without protection.
   - Do NOT report as SQL injection if using JdbcTemplate, prepared statements, ORMs, or other safe methods.
2. For other issues:
   - Report only high-confidence, visible problems in the given snippet.
   - Ignore undefined elements, code style, and minor optimizations.

Output Format (in Chinese):

1. SQL注入漏洞数量：[数字]

2. SQL注入漏洞详情（如果存在）：
位置：[行号]
描述：[如何未经保护的用户输入进入查询]
证据：[漏洞代码片段]
修复建议：[使用参数化查询或适当转义]

3. 其他高严重度问题：
位置：[行号]
类型：[bug/逻辑/安全/性能]
描述：[简洁的问题描述]
修复建议：[简短建议]

4. 高严重度问题总数（包括SQL注入）：[数字]

Do not include any other explanations or empty lines.

Code:
```[language]
[code]
```

这个优化版本：
1. 合并了SQL注入和其他问题的分析指令，使整体结构更简洁。
2. 保留了对SQL注入的特别关注，同时明确了判断标准。
3. 简化了输出格式，使其更加结构化和易于解析。
4. 保持了中文输出的要求。
5. 删除了冗余的解释和指示。
6. 增加了高严重度问题总数的统计，包括SQL注入在内。

这个版本应该能够引导LLM更准确地识别SQL注入问题，同时不影响对其他类型问题的正常分析和报告。它还应该能产生更一致和结构化的输出，便于后续处理和分析。
"""

review="""
根据您的要求，我理解您希望LLM在输出时使用带编号的多级列表，但目前LLM的输出中子列表项没有带编号。为了解决这个问题，我们可以进一步优化提示语，明确要求LLM在输出所有级别的列表项时都使用编号。以下是修改后的提示语：

```
Review for critical issues:
1. System crashes
2. Severe security vulnerabilities
3. Core functionality failures
4. Major performance problems

Ignore all commented-out code during review.

SQL injection guidelines:
- Only flag SQL injection risks when user input is directly concatenated into SQL strings without proper sanitization or parameterization.
- SQL statements constructed entirely from static strings or using parameterized queries are safe.
- Do not report SQL injection for static string concatenation or properly escaped/sanitized inputs.

Classify as non-critical:
- Minor performance optimizations
- Exception handling
- Potential SQL vulnerabilities in static strings or parameterized queries

Separate critical and non-critical issues. Show the number of each at the beginning.

Use numbered multi-level lists in the output. Ensure all levels of the list, including sub-items, are numbered.

Format:
Critical issues: [number]
1. [Issue description]
   1.1. [Sub-item if necessary]
   1.2. [Sub-item if necessary]
2. [Issue description]
   2.1. [Sub-item if necessary]
   2.2. [Sub-item if necessary]
3. [Issue description]
   3.1. [Sub-item if necessary]
   3.2. [Sub-item if necessary]

Non-critical issues: [number]
1. [Issue description]
   1.1. [Sub-item if necessary]
   1.2. [Sub-item if necessary]
2. [Issue description]
   2.1. [Sub-item if necessary]
   2.2. [Sub-item if necessary]
3. [Issue description]
   3.1. [Sub-item if necessary]
   3.2. [Sub-item if necessary]

Max 100 Chinese characters total for each section.

Provide only the requested information. No explanations or repetitions.
If no issues found, provide no output.

Code:
```
[Code here]
```
```

这个优化后的提示语：

1. 明确要求使用带编号的多级列表。
2. 提供了一个示例格式，展示了如何对所有级别的列表项进行编号。
3. 保留了之前的所有指导原则和限制。

这个修改应该能够引导LLM在输出时为所有级别的列表项添加编号，包括子列表项。这样可以确保输出的格式符合您的要求，即使用带编号的多级列表。

Citations:
[1] https://relevanceai.com/docs/tool/tool-steps/llms/llm-output
[2] https://www.cnblogs.com/gogoSandy/p/18264275
[3] https://www.ctyun.cn/developer/article/463376515866693
[4] https://blog.csdn.net/wtyuong/article/details/134556655
[5] https://api.python.langchain.com/en/latest/outputs/langchain_core.outputs.llm_result.LLMResult.html
"""


review="""
理解了您的需求。我们可以进一步优化LLM提示语，以减少这些情况下的SQL注入漏洞误报。以下是更新后的提示语：

```
Review for critical issues:
1. System crashes
2. Severe security vulnerabilities
3. Core functionality failures
4. Major performance problems

Ignore all commented-out code during review.

SQL injection guidelines:
- SQL statements constructed entirely from static strings (including multiple static strings concatenated together) are safe.
- Queries using JdbcTemplate, ORM frameworks (e.g., Hibernate, MyBatis), or other parameterized query methods are generally safe.
- Stored procedures are typically precompiled and safe from SQL injection.
- Dynamic SQL generation is often safe if proper parameterization or escaping is used.
- String concatenation followed by proper escaping or validation is safe.
- Usage of security-focused APIs or library functions that handle SQL injection protection is safe.
- Only flag SQL injection risks when user input or dynamic content is directly concatenated into SQL strings without any form of sanitization, parameterization, or security measures.

Classify as non-critical:
- Minor performance optimizations
- Exception handling
- Potential SQL vulnerabilities in static strings, parameterized queries, or when using secure frameworks/APIs

Separate critical and non-critical issues. Show the number of each at the beginning.

Use numbered multi-level lists in the output. Ensure all levels of the list, including sub-items, are numbered.

Format:
Critical issues: [number]
1. [Issue description]
   1.1. [Sub-item if necessary]
   1.2. [Sub-item if necessary]
2. [Issue description]
   2.1. [Sub-item if necessary]
   2.2. [Sub-item if necessary]

Non-critical issues: [number]
1. [Issue description]
   1.1. [Sub-item if necessary]
   1.2. [Sub-item if necessary]
2. [Issue description]
   2.1. [Sub-item if necessary]
   2.2. [Sub-item if necessary]

Max 100 Chinese characters total for each section.

Provide only the requested information. No explanations or repetitions.
If no issues found, provide no output.

Code:
```
[Code here]
```
```

这个优化后的提示语：

1. 明确指出ORM框架（如Hibernate、MyBatis）的查询通常是安全的。
2. 说明存储过程通常是预编译的，不易受到SQL注入攻击。
3. 指出动态SQL生成如果使用了适当的参数化或转义，通常是安全的。
4. 强调字符串拼接后如果进行了适当的转义或验证，也是安全的。
5. 提到使用专注于安全性的API或库函数处理SQL注入保护是安全的。
6. 明确指出只有当用户输入或动态内容直接拼接到SQL字符串中，且没有任何形式的清理、参数化或安全措施时，才应标记为SQL注入风险。

这个修改应该能够帮助LLM更准确地识别真正的SQL注入风险，大大减少在您提到的情况下的误报。同时，它仍然保持对可能存在的实际SQL注入风险的警惕。
"""

review="""
理解了您的需求。我们需要确保LLM尽可能发现所有的bug或问题，同时减少误报。以下是进一步优化的系统提示语和用户提示语：

### 系统提示语

```
You are a code review assistant. Your goal is to identify all potential bugs or issues in the code while minimizing false positives. Focus on:

1. Critical issues:
   - System crashes
   - Severe security vulnerabilities (including SQL injection)
   - Core functionality failures
   - Major performance problems

2. SQL injection safety:
   - Safe: Static SQL, parameterized queries, ORM frameworks (e.g., Hibernate, MyBatis), stored procedures, proper escaping
   - Unsafe: Direct user input concatenation without sanitization

3. Non-critical issues:
   - Minor optimizations
   - Exception handling
   - Potential vulnerabilities in safe SQL practices

Ignore commented code. Be concise and accurate. Prioritize accuracy over quantity.
```

### 用户提示语

```
Review the code enclosed between triple backticks (```):

```
[Code here]
```

Report using numbered lists:

Critical issues: [number]
1. [Brief description]
   1.1. [Sub-item if needed]

Non-critical issues: [number]
1. [Brief description]
   1.1. [Sub-item if needed]

Max 100 Chinese characters per section. No explanations. If no issues, provide no output.
```

这个优化后的提示语：

1. **系统提示语**:
   - 明确指出目标是尽可能发现所有的bug或问题，同时减少误报。
   - 强调准确性优先于数量。
   - 保持了对SQL注入安全性的详细指导，并明确哪些情况是安全的，哪些是不安全的。
   - 保留了对非严重问题的分类。

2. **用户提示语**:
   - 明确指出代码是用三个反引号（```）包裹的。
   - 保持代码部分在前面，让LLM首先关注要审查的代码。
   - 提供了简明的输出格式指示。
   - 保持了字符限制和不输出解释的要求。

这个版本的提示语应该能够帮助LLM更准确地识别和分析代码中的问题，同时减少误报的可能性。它明确了目标和优先级，有助于LLM在进行代码审查时保持高效和准确。
"""


code_biz_desc="""
You're a business analyst. Describe the core business function of the code enclosed in triple backticks (```). 

**Instructions:**
1. Focus exclusively on business purpose
2. Avoid any technical details

code to analysis:
```java
code
```

**Output Requirements:**
1. Output answer directly
2. Don't repeat or explain the question
3. Use Chinese
4. 1-2 sentences
5. Max 20 characters

当然，这里是优化后的完整提示语：

```
You're a business analyst. Describe the core business function of the code enclosed in triple backticks (```). 

**Instructions:**
1. Focus on business purpose only
2. No technical details

Code to analyze:
```java
code
```

**Output Requirements:**
1. Output answer directly
2. Don't repeat or explain the question
3. Use Chinese
4. 1-2 sentences
5. Max 20 characters
```

这个优化版本保持了所有关键要求，同时简化了语言，使其更加简洁明了。

"""