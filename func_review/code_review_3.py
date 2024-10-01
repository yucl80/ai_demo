
from openai import OpenAI

# 设置API密钥
api_key = "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"
base_url = "http://127.0.0.1:8000/v1/"



from groq import Groq

# client = Groq( api_key=api_key)
client = OpenAI(base_url=base_url, api_key=api_key)

def initial_code_review(code_snippet, language="Python", context=""):
    # First prompt for initial review
    prompt = f"""
    Review the following {language} code. Identify logical errors, security vulnerabilities, and performance improvements. 
    Context: {context}.
    
    Code:
    {code_snippet}
    
    Instructions:
    - Check for syntax and logical errors.
    - Identify security vulnerabilities.
    - Suggest performance optimizations.
    - Evaluate readability and adherence to coding standards.
    
    Output rules:
    1. Only ouput the issues ,no suggests
    2. Use Chinese
    """

    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role":"user","content": prompt}],
        max_tokens=500,
        temperature=0.5,
        timeout=10000
    )

    return response.choices[0].message.content

def recheck_review(initial_feedback, code_snippet):
    # Second prompt for re-evaluation
    prompt = f"""
    Re-evaluate the following initial code review feedback for the given code snippet. 
    Ensure that all identified issues are accurate and provide any additional insights if necessary.
    
    Code:
    {code_snippet}
    
    Initial Feedback:
    {initial_feedback}
    
    Instructions:
    - Verify the accuracy of the initial feedback.
    - Provide additional insights or corrections if needed.
    - Ensure completeness of the review.
    
    Output rules:
    1. Only ouput the issues ,no suggests
    2. Use Chinese
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content": prompt}],
        max_tokens=500,
        temperature=0.5,
        timeout=10000
    )

    return response.choices[0].message.content

# Example usage
code_snippet = """
def calculate_factorial(n):
    if n == 0:
        return 1
    else:
        return n * calculate_factorial(n - 1)
"""

context = "This function is used in a high-performance computing application."

# Step 1: Initial Review
initial_feedback = initial_code_review(code_snippet, language="Python", context=context)

print(initial_feedback)

# Step 2: Re-evaluation
final_feedback = recheck_review(initial_feedback, code_snippet)

# Output the final review
print(final_feedback)
