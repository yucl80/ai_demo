from sentence_transformers import SentenceTransformer
from text2vec import  cos_sim

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
sentences = ['search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten']
embeddings = model.encode(sentences)
# print(embeddings)


sentences = ['search_query: Who is Laurens van Der Maaten?']
embeddings2 = model.encode(sentences)
# print(embeddings2)

print(embeddings @ embeddings2.T)

print(cos_sim(embeddings, embeddings2))

