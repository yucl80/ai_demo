import time
from FlagEmbedding import BGEM3FlagModel
from langchain_community.utils.math import cosine_similarity_top_k 

model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.","adfsadfasdfsdf","asdfs asdfasd", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_1 = model.encode(sentences_1, 
                            batch_size=4, 
                            max_length=64, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
st = time.perf_counter()

embeddings_2 = model.encode(sentences_2)['dense_vecs']
et = time.perf_counter() - st
print("emb time ：", et)
#similarity = embeddings_1 @ embeddings_2.T
#print(similarity)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]

