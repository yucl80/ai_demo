from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings

import os
os.environ["OPENAI_API_KEY"]="NOKEY" 
client = OpenAI(base_url="http://127.0.0.1:8000/v1/")


# from yucl.utils import HttpEmbeddingsClient
#embeddings = OpenAIEmbeddings(model="bge-large-zh-v1.5",base_url="http://localhost:8000/v1",api_key="get_api_key")
# 
# embeddings = HttpEmbeddingsClient(model="bge-m3")
# result=embeddings.embed_documents(["你好，世界！"])
# print(len(result[0]))

# emb = client.embeddings.create(input=["hello"],model="bge-large-zh-v1.5",encoding_format="float")
# print(emb)

def get_embedding(text, model="bge-m3"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

from text2vec import SentenceModel, cos_sim, semantic_search, Similarity, EncoderType
import torch


corpus = ["我爱你", "我恨你" ,"我喜欢你","我讨厌你","我不喜欢你","我不爱你","I love you","I like you","I hate you","I don't like you"]

import time
begin_time = time.time()
response = client.embeddings.create(input=corpus, model="bge-m3").data[0].embedding
end_time = time.time()
print("Time taken  {} seconds".format( end_time - begin_time))

corpus_embeddings = []
for str in corpus:   
    response = get_embedding(str)  
    corpus_embeddings.append(torch.tensor(response))
   
query_embedding = get_embedding("我爱你") 

hits = semantic_search(torch.tensor(query_embedding), corpus_embeddings, top_k=5)
hits = hits[0]      #Get the hits for the first query
for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
       
