import requests


def get_weather_data(coordinates):
    """
    Fetches weather data from the Open-Meteo API for the given latitude and longitude.

    Args:
    coordinates (tuple): The latitude of the location.

    Returns:
    float: The current temperature in the coordinates you've asked for
    """
    latitude, longitude = coordinates
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,wind_speed_10m",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        print(
            f"""Temperature in C: {response.json()["current"]["temperature_2m"]}""")
    else:
        return {"error": "Failed to fetch data, status code: {}".format(response.status_code)}


def get_coordinates_from_city(city_name):
    """
    Fetches the latitude and longitude of a given city name using the Maps.co Geocoding API.

    Args:
    city_name (str): The name of the city.

    Returns:
    tuple: The latitude and longitude of the city.
    """
    base_url = "https://geocode.maps.co/search"
    params = {"q": city_name}

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            # Assuming the first result is the most relevant
            return data[0]["lat"], data[0]["lon"]
        else:
            return {"error": "No data found for the given city name."}
    else:
        return {"error": "Failed to fetch data, status code: {}".format(response.status_code)}


RAVEN_PROMPT = \
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

QUESTION = "Whats's the weather like in Seattle right now?"

# Now, let's prompt Raven!
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
headers = {
        "Content-Type": "application/json"
}

def query(payload):
	"""
	Sends a payload to a TGI endpoint.
	"""
	import requests
	response = requests.post(API_URL, headers=headers, json=payload)
   
	return response.json()

def query_raven(prompt):
	"""
	This function sends a request to the TGI endpoint to get Raven's function call.
	This will not generate Raven's justification and reasoning for the call, to save on latency.
	"""
	import requests
	output = query({
		"inputs": prompt,
		"parameters" : {"temperature" : 0.001, "stop" : ["<bot_end>"], "do_sample" : False, "max_new_tokens" : 2000, "return_full_text" : False}})
	call = output[0]["generated_text"].replace("Call:", "").strip()
	return call

def query_raven_with_reasoning(prompt):
	"""
	This function sends a request to the TGI endpoint to get Raven's function call AND justification for the call
	"""
	import requests
	output = query({
		"inputs": prompt,
		"parameters" : {"temperature" : 0.001, "do_sample" : False, "max_new_tokens" : 2000, "return_full_text" : False}})
	call = output[0]["generated_text"].replace("Call:", "").strip()
	return call

my_question = RAVEN_PROMPT.format(query = QUESTION)
raven_call = query_raven(my_question)
print (f"Raven's Call: {raven_call}")

exec(raven_call)

my_question = RAVEN_PROMPT.format(query = QUESTION)
raven_call = query_raven_with_reasoning(my_question)
print (f"Raven's Call: {raven_call}")

