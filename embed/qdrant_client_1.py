from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://486d3147-d00f-48a8-b03a-d7754b0f1860.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="FEg7NEaqJe14DYxmPmo0h60p-I_Dw8L9LSsbubipGpKJymSG9Kr0VA",
)

print(qdrant_client.get_collections())