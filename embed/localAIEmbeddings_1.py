from langchain_community.embeddings import LocalAIEmbeddings
embeddings = LocalAIEmbeddings(
    openai_api_base="http://localhost:8000", model="bge-large-zh-v1.5",openai_api_key="YOUR_API_KEY"
)
text = "This is a test document."
query_result = embeddings.embed_query(text)