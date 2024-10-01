from openai import OpenAI


api_key = "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"
base_url = "http://127.0.0.1:8000/v1/"



from groq import Groq

# client = Groq( api_key=api_key)
client = OpenAI(base_url=base_url, api_key=api_key)

def review_code(code_snippet, language="Python", context=""):
    # Comprehensive prompt for a single LLM call
    prompt = f"""
    You are a code review assistant tasked with reviewing a {language} code snippet.
    
    Context: The code is part of a high-performance computing application. It is crucial to ensure that the code is efficient and secure.
    
    Code:
    {code_snippet}
    
    Instructions:
    - Identify any syntax or logical errors, providing specific examples.
    - Highlight potential security vulnerabilities with detailed explanations.
    - Suggest performance optimizations, focusing on computational efficiency.
    - Evaluate readability and adherence to coding standards, offering examples of improvements.
    - Break down the review process into logical steps, providing reasoning for each identified issue.
    - Ensure the review is comprehensive and contextually relevant to the application's requirements.
    - Please do not provide modification suggestions or examples; only list the identified issues.
    
    Output rules:
    1. Output the issues detail only, no suggests
    2. Use Chinese
    """
    
    # Single LLM call for comprehensive review
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role":"user","content": prompt}],
        max_tokens=500,
        temperature=0.2,
        top_p=0.1
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
import time
begin_time = time.time()
final_review = review_code(code_snippet, language="Python", context=context)
print(f"used time {time.time()-begin_time}")

# Output the final review result
print(final_review)
