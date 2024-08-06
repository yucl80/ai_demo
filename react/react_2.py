from openai import OpenAI
import re

# Set your OpenAI API key
client = OpenAI(base_url="http://127.0.0.1:8000/v1/", api_key= "your-api-key")

class Tool:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def use(self, *args):
        return self.func(*args)

def search_tool(query):
    # Simulated search function
    if "2024 Olympics" in query:
        return "The host city for the 2024 Olympics is Paris."
    return "No relevant information found."

def distance_calculator(city1, city2):
    # Simulated distance calculator
    if (city1.lower() == "new york" and city2.lower() == "paris") or \
       (city1.lower() == "paris" and city2.lower() == "new york"):
        return "5837 kilometers"
    return "Unable to calculate distance."

class AIAssistant:
    def __init__(self):
        self.tools = {
            "Search": Tool("Search", search_tool),
            "Distance Calculator": Tool("Distance Calculator", distance_calculator)
        }

    def think(self, task):
        print("Thought:", task)

    def act(self, action, *args):
        print(f"Action: {action}")
        if action.startswith("Use Tool"):
            tool_name = re.search(r"Use Tool\s*:\s*(\w+)", action).group(1)
            tool = self.tools.get(tool_name)
            if tool:
                result = tool.use(*args)
                print(f"Observation: {result}")
                return result
        elif action.startswith("Search"):
            result = self.tools["Search"].use(args[0])
            print(f"Observation: {result}")
            return result
        else:
            print("Unable to perform this action")

    def answer(self, answer):
        print("Answer:", answer)

    def process_task(self, task):
        self.think("I need to find the host city for the 2024 Olympics and calculate the flight distance from New York to that city.")
        
        search_result = self.act("Search: 2024 Olympics host city")
        host_city = "Paris"  # Extract from search result
        
        self.think(f"Now I need to calculate the flight distance from New York to {host_city}.")
        
        distance = self.act("Use Tool: Distance Calculator", "New York", host_city)
        
        final_answer = f"The host city for the 2024 Olympics is {host_city}. The flight distance from New York to {host_city} is approximately {distance}."
        self.answer(final_answer)

def openai_chat_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def main():
    assistant = AIAssistant()
    task = "Please find the host city for the 2024 Olympics and calculate the flight distance from New York to that city."
    assistant.process_task(task)

    # Example usage of OpenAI API for chat completion
    prompt = "Please find the host city for the 2024 Olympics and calculate the flight distance from New York to that city."
    result = openai_chat_completion(prompt)
    print("OpenAI API Response:", result)

if __name__ == "__main__":
    main()