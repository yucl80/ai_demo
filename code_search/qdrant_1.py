from qdrant_client import QdrantClient
# qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance, for testing, CI/CD
# OR
# client = QdrantClient("http://192.168.85.130:6333")  # Persists changes to disk, fast prototyping



client = QdrantClient(host="192.168.85.130", grpc_port=6334, prefer_grpc=True)

docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]
metadata = [
    {"source": "Langchain-docs"},
    {"source": "Linkedin-docs"},
]
ids = [42, 2]

# Use the new add method
client.add(
    collection_name="demo_collection",
    documents=docs,
    metadata=metadata,
    ids=ids
)

search_result = client.query(
    collection_name="demo_collection",
    query_text="This is a query document"
)
print(search_result)