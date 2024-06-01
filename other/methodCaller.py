import inspect
from typing import Any, Callable, Dict
import json

import llama_cpp
from llama_cpp import LlamaGrammar

class MethodCaller:

    def __init__(self, model: Any):
        self.grammar = LlamaGrammar.from_file("/home/test/src/ai_demo/other/json.gbnf")
        self.model = model

    def get_schema(self, item: Callable) -> Dict[str, Any]:
        schema = {
            "name": item.__name__,
            "description": str(inspect.getdoc(item)),
            "signature": str(inspect.signature(item)),
            "output": str(inspect.signature(item).return_annotation),
        }
        return schema
    
    def run(self, query: str, function: Callable):
        
        schema = self.get_schema(function)
         
        prompt = f"""
            You are a helpful assistant designed to output JSON.
            Given the following function schema
            << {schema} >>
            and query
            << {query} >>
            extract the parameters values from the query, in a valid JSON format.
            Example:
            Input:
            query: "How is the weather in Hawaii right now in International units?"
            schema:
            {{
                "name": "get_weather",
                "description": "Useful to get the weather in a specific location",
                "signature": "(location: str, degree: str) -> str",
                "output": "<class 'str'>",
            }}

            Result: {{
                "location": "London",
                "degree": "Celsius",
            }}

            Input:
            query: {query}
            schema: {schema}
            Result:
        """

        completion = self.model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.1,
            max_tokens=500,
            grammar=self.grammar,
            stream=False,
        )

        output = completion["choices"][0]["message"]["content"]
        output = output.replace("'", '"').strip().rstrip(",")
        
        function_inputs = json.loads(output)
        print( function_inputs )
        return  function(**function_inputs)


# Function to be called described using the Python docstring format   
def add_two_numbers( first_number: int, second_number: int ) -> str:
    """
    Adds two numbers together.

    :param first_number: The first number to add.
    :type first_number: int

    :param second_number: The second number to add.
    :type second_number: int

    :return: The sum of the two numbers.
    """
    return first_number + second_number 


model = llama_cpp.Llama(
    model_path="/home/test/llm-models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=True,
)

fun = MethodCaller( model )

result = fun.run("Add 2 and 5", add_two_numbers)
print(result)
import time
begin = time.time()
result = fun.run("Add 66 and seventy", add_two_numbers)
endtime = time.time()
print("used time:", endtime - begin)
print(result)