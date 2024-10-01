# https://qdrant.tech/documentation/tutorials/code-search/
import json

structures = []
with open("structures.jsonl", "r") as fp:
    for i, row in enumerate(fp):
        entry = json.loads(row)
        structures.append(entry)

import inflection
import re

from typing import Dict, Any

def textify(chunk: Dict[str, Any]) -> str:
    # Get rid of all the camel case / snake case
    # - inflection.underscore changes the camel case to snake case
    # - inflection.humanize converts the snake case to human readable form
    name = inflection.humanize(inflection.underscore(chunk["name"]))
    signature = inflection.humanize(inflection.underscore(chunk["signature"]))

    # Check if docstring is provided
    docstring = ""
    if chunk["docstring"]:
        docstring = f"that does {chunk['docstring']} "

    # Extract the location of that snippet of code
    context = (
        f"module {chunk['context']['module']} "
        f"file {chunk['context']['file_name']}"
    )
    if chunk["context"]["struct_name"]:
        struct_name = inflection.humanize(
            inflection.underscore(chunk["context"]["struct_name"])
        )
        context = f"defined in struct {struct_name} {context}"

    # Combine all the bits and pieces together
    text_representation = (
        f"{chunk['code_type']} {name} "
        f"{docstring}"
        f"defined as {signature} "
        f"{context}"
    )

    # Remove any special characters and concatenate the tokens
    tokens = re.split(r"\W", text_representation)
    tokens = filter(lambda x: x, tokens)
    return " ".join(tokens)

text_representations = list(map(textify, structures))

from sentence_transformers import SentenceTransformer

nlp_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp_embeddings = nlp_model.encode(
    text_representations, show_progress_bar=True,
)

# Extract the code snippets from the structures to a separate list
code_snippets = [
    structure["context"]["snippet"] for structure in structures
]

code_model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-code",   
    trust_remote_code=True
)
code_model.max_seq_length = 8192  # increase the context length window
code_embeddings = code_model.encode(
    code_snippets, batch_size=4, show_progress_bar=True,
)

QDRANT_URL = "https://192.168.85.132:6333" # http://localhost:6333 for local instance
QDRANT_API_KEY = "THIS_IS_YOUR_API_KEY" # None for local instance

from qdrant_client import QdrantClient, models

client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
client.create_collection(
    "qdrant-sources",
    vectors_config={
        "text": models.VectorParams(
            size=nlp_embeddings.shape[1],
            distance=models.Distance.COSINE,
        ),
        "code": models.VectorParams(
            size=code_embeddings.shape[1],
            distance=models.Distance.COSINE,
        ),
    }
)

import uuid

points = [
    models.PointStruct(
        id=uuid.uuid4().hex,
        vector={
            "text": text_embedding,
            "code": code_embedding,
        },
        payload=structure,
    )
    for text_embedding, code_embedding, structure in zip(nlp_embeddings, code_embeddings, structures)
]

client.upload_points("qdrant-sources", points=points, batch_size=64)

query = "How do I count points in a collection?"

hits = client.query_points(
    "qdrant-sources",
    query=nlp_model.encode(query).tolist(),
    using="text",
    limit=5,
).points

hits = client.query_points(
    "qdrant-sources",
    query=code_model.encode(query).tolist(),
    using="code",
    limit=5,
).points

responses = client.query_batch_points(
    "qdrant-sources",
    requests=[
        models.QueryRequest(
            query=nlp_model.encode(query).tolist(),
            using="text",
            with_payload=True,
            limit=5,
        ),
        models.QueryRequest(
            query=code_model.encode(query).tolist(),
            using="code",
            with_payload=True,
            limit=5,
        ),
    ]
)

results = [response.points for response in responses]

results = client.search_groups(
    "qdrant-sources",
    query_vector=(
        "code", code_model.encode(query).tolist()
    ),
    group_by="context.module",
    limit=5,
    group_size=1,
)
