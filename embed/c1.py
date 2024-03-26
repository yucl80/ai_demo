from langchain_openai import OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings
import time
import os

embeddings = OpenAIEmbeddings(base_url = "http://127.0.0.1:8000/v1/",api_key="APIKEY")

st = time.perf_counter()
#embed = embeddings.embed_documents(["texts"],10)

from openai import OpenAI
client = OpenAI(base_url = "http://127.0.0.1:8000/v1/",api_key="APIKEY")

def get_embedding(text, model="text"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

r = get_embedding("test txt", model='text-embedding-3-small')
et = time.perf_counter()
print(et - st)
print(len(r))