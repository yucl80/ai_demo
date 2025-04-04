from semantic_router import Route

# we could use this as a guide for our chatbot to avoid political conversations
politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

# we place both of our decisions together into single list
routes = [politics, chitchat]

import os
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

# for Cohere
# os.environ["COHERE_API_KEY"] = "<YOUR_API_KEY>"
# encoder = CohereEncoder()

# or for OpenAI
# os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
encoder = OpenAIEncoder("model",openai_api_key="nokey",openai_base_url="http://localhost:8000/v1/")

from semantic_router.routers import SemanticRouter

rl = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

rl("don't you love politics?").name