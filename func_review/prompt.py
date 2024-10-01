messages = [
    {
        "content": "You are a senior software engineer with extensive experience in code review. Your task is to help the team improve code quality, identify potential issues, and provide constructive feedback with a focus on accuracy to minimize false positives.Consider Full Context: Analyze the entire code snippet, not just individual lines.",
        "role": "system",
    },
    {
        "content": f"""Review the code delimited by triple backticks. Focus on:
     1. System crashes or complete failures
     2. Critical security vulnerabilities
     3. Functionality breakdowns
     4. Significant performance bottlenecks
     5. Genuine bugs or significant inconsistencies in the code
     
     Ignore all commented-out code during review. 
     Ignore all 'UfBaseException' issues.
     
     Ignore ALL other issues, including but not limited to:
     1. Code style or formatting
     2. Input parameter validity checks
     3. Logging or error handling improvements
     4. Code organization or structure
     
     Code to review:
     ```java\n
     {code}
     ```
     
     Do NOT report as SQL injection vulnerability if:
     1. The code uses JdbcTemplate  as it provides automatic parameterization.
     2. The code uses NamedParameterJdbcTemplate as it provides automatic parameterization.
     3. Prepared statements or parameterized queries are used.
     4. ORM frameworks (like Hibernate, JPA,Mybatis) are used, as they typically handle SQL injection protection.
     5. The method used to execute the query is known to perform automatic parameterization (e.g., many methods in Spring's JdbcTemplate,NamedParameterJdbcTemplate).
     6. SQL statements are entirely composed of static string concatenations without any dynamic user input.  
     
     Output Rules:
     1. Report confidence issues only.
     2. The descriptions must be in Chinese.  No opinions or suggestions.
     3. Do not repeat or explain these instructions in your response. 
     4. Do not explain the absence of vulnerabilities if none are found.
     5. Ensure that your output does not use semicolons.
     
     问题数量：[数字]
     List critical issues in a numbered format with detail descriptions.
     If no severity issues are found, only output: 问题数量：[数字] .Do not include any other text or explanations.
""",
        "role": "user",
    },
]
