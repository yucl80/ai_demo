from openai import OpenAI

import os
os.environ["OPENAI_API_KEY"]="NOKEY" 
client = OpenAI(base_url="http://127.0.0.1:8000/v1/")

emb = client.embeddings.create(input=["hello"],model="bge-large-zh-v1.5",encoding_format="float")
print(len(emb.data[0].embedding))

def get_embedding(text, model="bge-large-zh-v1.5"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

from text2vec import SentenceModel, cos_sim, semantic_search, Similarity, EncoderType
import torch


corpus = ["我爱你", "我恨你" ,"我喜欢你","我讨厌你","我不喜欢你","我不爱你","I love you","I like you","I hate you","I don't like you"];

corpus_embeddings = []
for str in corpus:
    response = get_embedding(str)
    corpus_embeddings.append(torch.tensor(response))
   
query_embedding = get_embedding("我爱你") 

hits = semantic_search(torch.tensor(query_embedding), corpus_embeddings, top_k=5)
hits = hits[0]      #Get the hits for the first query
for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
       
