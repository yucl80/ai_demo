from openai import OpenAI


api_key = "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"
base_url = "http://127.0.0.1:8000/v1/"
from collections import Counter

# Set your OpenAI API key

client = OpenAI(base_url=base_url, api_key=api_key)

def generate_review(code_snippet, language="Python", context="", num_iterations=5):
    # Detailed prompt to ensure clarity and specificity, requesting Markdown format
    prompt = f"""
    You are a code review assistant tasked with reviewing the following {language} code snippet.
    
    Context: The code is part of a high-performance computing application. It is crucial to ensure that the code is efficient and secure. Remember this context for consistency in your responses.

    Code:
    ```{language}
    {code_snippet}
    ```

    Instructions:
    - Identify any syntax errors or logical errors and provide detailed problem descriptions.
    - Highlight potential security vulnerabilities, giving detailed problem descriptions.
    - Evaluate readability and adherence to coding standards, providing problem descriptions.
    - Please do not provide modification suggestions or examples; only list the identified issues.
    - Ensure that your output is consistent across multiple iterations.
    - Format your output using Markdown syntax.
    - Cache important details from previous outputs to maintain consistency in your responses.
    """

    # Generate multiple outputs for self-consistency
    reviews = []
    for _ in range(num_iterations):
        response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role":"user","content": prompt}],
        max_tokens=500,
        temperature=0.2,
        top_p=0.1
    )
        reviews.append(response.choices[0].message.content)
    
    # Use Counter to find the most common output
    most_common_review, _ = Counter(reviews).most_common(1)[0]
    return most_common_review

# Example usage
code_snippet = """
def calculate_factorial(n):
    if n == 0:
        return 1
    else:
        return n * calculate_factorial(n - 1)
"""

context = "This function is used in a high-performance computing application."

# Perform the code review with optimized settings
final_review = generate_review(code_snippet, language="Python", context=context)

# Output the final review result
print(final_review)
