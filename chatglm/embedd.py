from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer, util
import torch
from code.deploy.yucl_utils.jwt_token import get_api_key

client = ZhipuAI(api_key=get_api_key()) 
corpus = ["我爱你", "我恨你" ,"我喜欢你","我讨厌你","我不喜欢你","我不爱你","I love you","I like you","I hate you","I don't like you"];

corpus_embeddings = []
for str in corpus:
    response = client.embeddings.create(
        model="embedding-2", #填写需要调用的模型名称
        input= str,
    )
    corpus_embeddings.append(torch.tensor(response.data[0].embedding))
   
query_embedding = client.embeddings.create(
    model="embedding-2", #填写需要调用的模型名称
    input= "我爱你",
).data[0].embedding

hits = util.semantic_search(torch.tensor(query_embedding), corpus_embeddings, top_k=5)
hits = hits[0]      #Get the hits for the first query
for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
       