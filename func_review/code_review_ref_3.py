from openai import OpenAI


api_key = "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"
base_url = "http://127.0.0.1:8000/v1/"



from groq import Groq

# client = Groq( api_key=api_key)
client = OpenAI(base_url=base_url, api_key=api_key)


from collections import Counter



def generate_review(code_snippet, language="Python", context="", num_iterations=5):
    # Detailed prompt to ensure clarity and specificity
    prompt = f"""
    You are a code review assistant tasked with reviewing the following {language} code snippet. 
       
    Context: The code is part of a high-performance computing application. It is crucial to ensure that the code is efficient and secure. Remember this context for consistency in your responses.
    
    Code:
    {code_snippet}  
    
    Instructions:    
    - Identify any logical errors and provide detailed problem descriptions.
    - Highlight potential security vulnerabilities, giving detailed problem descriptions.
    - Evaluate readability and adherence to coding standards, providing problem descriptions.
    - Please do not provide modification suggestions or examples; only list the identified issues.
    - Ensure that your output is consistent across multiple iterations.    
    - Cache important details from previous outputs to maintain consistency in your responses.

    
    Output rules:   
    1. Use Chinese
    2. Format your output using Markdown syntax.
    """

    # Generate multiple outputs for self-consistency
    reviews = []
      # Single LLM call for comprehensive review
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role":"user","content": prompt}],
        max_tokens=500,
        temperature=0.2,
        top_p=0.1
    )
    
    reviews.append( response.choices[0].message.content)
    
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
import time
begin_time = time.time()

final_review = generate_review(code_snippet, language="Python", context=context)
print(f"used time {time.time()-begin_time}")
# Output the final review result
print(final_review)
