from llama_cpp import Llama


# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path="/home/test/llm-models/nexusraven-v2-13b.Q4_K_M.gguf",  # Download the model file first
  n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=12,            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=0         # The number of layers to offload to GPU, if you have GPU acceleration available
)

prompt_template = \
'''
Function:
def get_weather_data(coordinates):
    """
    Fetches weather data from the Open-Meteo API for the given latitude and longitude.

    Args:
    coordinates (tuple): The latitude of the location.

    Returns:
    float: The current temperature in the coordinates you've asked for
    """

Function:
def get_coordinates_from_city(city_name):
    """
    Fetches the latitude and longitude of a given city name using the Maps.co Geocoding API.

    Args:
    city_name (str): The name of the city.

    Returns:
    tuple: The latitude and longitude of the city.
    """

User Query: {query}<human_end>

'''

prompt = prompt_template.format(query="What's the weather like in Seattle right now?")

# Simple inference example
output = llm(
 # "Function:\ndef function_here(arg1):\n  """\n    Comments explaining the function here\n\n    Args:\n    list args\n\n    Returns:\n    list returns\n    """\n\nFunction:\ndef another_function_here(arg1):\n  ...\n\nUser Query: {prompt}<human_end>", # Prompt
  prompt,
  max_tokens=512,  # Generate up to 512 tokens
  stop=["<bot_end>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
 # stop=["</s>"],
  echo=False,        # Whether to echo the prompt
  temperature=0.001,
)

print(output)

import time
begin = time.time()
prompt = prompt_template.format(query="What's the weather like in New York right now?")

# Simple inference example
output = llm(
 # "Function:\ndef function_here(arg1):\n  """\n    Comments explaining the function here\n\n    Args:\n    list args\n\n    Returns:\n    list returns\n    """\n\nFunction:\ndef another_function_here(arg1):\n  ...\n\nUser Query: {prompt}<human_end>", # Prompt
  prompt,
  max_tokens=512,  # Generate up to 512 tokens
  stop=["<bot_end>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
 # stop=["</s>"],
  echo=False,        # Whether to echo the prompt
  temperature=0.001,
)
end = time.time()
print("Time taken:", end - begin)

# Chat Completion API

# llm = Llama(model_path="/home/test/llm-models/nexusraven-v2-13b.Q4_K_M.gguf", chat_format="llama-2")  # Set chat_format according to the model you are using
# output = llm.create_chat_completion(
#     messages = [
#         {"role": "system", "content": "You are a story writing assistant."},
#         {
#             "role": "user",
#             "content": "Write a story about llamas."
#         }
#     ]
# )
# print(output)
