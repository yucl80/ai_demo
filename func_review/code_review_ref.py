from openai import OpenAI


api_key = "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"
base_url = "http://127.0.0.1:8000/v1/"



from groq import Groq

# client = Groq( api_key=api_key)
client = OpenAI(base_url=base_url, api_key=api_key)

def review_code(code_snippet, language="Python", context=""):
    # Comprehensive prompt for a single LLM call
    prompt = f"""
    You are a code review assistant. Review the following {language} code snippet with the given context.
    
    Code:
    {code_snippet}
    
    Context: {context}
    
    Instructions:
    - Identify any syntax or logical errors.
    - Highlight potential security vulnerabilities.
    - Suggest performance optimizations.
    - Evaluate readability and adherence to coding standards.
    - Provide actionable feedback with specific examples or suggestions for improvement.
    - Ensure the review is accurate, comprehensive, and contextually relevant.
    
    Output rules:
    1. Only ouput the issues detail ,no suggests
    2. Use Chinese
    """
    
    # Single LLM call for comprehensive review
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role":"user","content": prompt}],
        max_tokens=500,
        temperature=0.5
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

# Perform a single comprehensive review
final_review = review_code(code_snippet, language="Python", context=context)

# Output the final review result
print(final_review)
