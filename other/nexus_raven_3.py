from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
   base_url="http://localhost:8000",  # Set the URL of the OpenAI API endpoint
   api_key="YOUR_API_KEY_HERE",  # Set your OpenAI API key here  
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

prompt = prompt_template.format(query="What's the weather like in Seattle  and  Shanghai right now?")


output =llm.invoke(prompt, max_tokens=200, stop=["<bot_end>"])  # Invoke the API and print the output

print(output)

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
