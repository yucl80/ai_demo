import numpy as np
import faiss
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1/", api_key="api_key")
# Initialize Faiss index
dimension = 512  # Example dimension size
index = faiss.IndexFlatL2(dimension)

# Example entity vectors and their associated information
entities = {
    "New York": {
        "vector": np.random.rand(dimension).astype('float32'),
        "type": "city",
        "country": "USA",
        "population": "8.4 million",
        "known_for": ["Statue of Liberty", "Central Park", "Broadway"]
    },
    "San Francisco": {
        "vector": np.random.rand(dimension).astype('float32'),
        "type": "city",
        "country": "USA",
        "population": "0.9 million",
        "known_for": ["Golden Gate Bridge", "Silicon Valley", "Cable Cars"]
    },
    "Nobel Prize": {
        "vector": np.random.rand(dimension).astype('float32'),
        "type": "award",
        "categories": ["Physics", "Chemistry", "Medicine", "Literature", "Peace", "Economics"],
        "first_awarded": 1901,
        "notable_winners": ["Albert Einstein", "Marie Curie", "Martin Luther King Jr."]
    }
}

# Add entity vectors to the index
for entity, data in entities.items():
    index.add(np.array([data["vector"]]))

def get_entity_info(entity: str) -> dict:
    return entities.get(entity, {})

def extract_entities(query: str) -> list:
    """
    Use LLM to extract entities from the user query.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with extracting named entities from a given query. Please return the entities as a comma-separated list."},
            {"role": "user", "content": f"Extract the named entities from the following query: {query}"}
        ],
        max_tokens=100
    )
    entities = response.choices[0].message.content.split(',')
    return [entity.strip() for entity in entities if entity.strip()]

def search_entities(query: str, k: int = 3) -> list:
    extracted_entities = extract_entities(query)
    print(f"Extracted entities: {extracted_entities}")  # For debugging
    
    closest_entities = []
    for entity in extracted_entities:
        # Generate vector for the entity (this is a placeholder, replace with actual embedding generation)
        entity_vector = np.random.rand(dimension).astype('float32')
        D, I = index.search(np.array([entity_vector]), k)
        closest_entities.extend([list(entities.keys())[i] for i in I[0]])
    
    # Remove duplicates and limit to k entities
    return list(dict.fromkeys(closest_entities))[:k]