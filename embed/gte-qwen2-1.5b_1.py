from sentence_transformers import SentenceTransformer
import torch
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)

# 对全连接层进行动态量化
model = torch.quantization.quantize_dynamic(
    model,  # 要量化的模型
    {torch.nn.Linear},  # 量化的层类型（这里只对Linear层量化）
    dtype=torch.qint8  # 使用8位量化
)

# In case you want to reduce the maximum length:
model.max_seq_length = 8192

queries = [
    "how much protein should a female eat",
    "summit define",
]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]

query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

scores = (query_embeddings @ document_embeddings.T) * 100
print(scores.tolist())

embeddings = model.encode([    
    'def f(a,b): if a>b: return a else return b',
    "def f(x,y): if x<y: return y else return x"
])
queries = ['get the maximum value']
import time
begin = time.time()
query_embeddings = model.encode(queries, prompt_name="query")
end = time.time()
print("Time elapsed: ", end - begin)
print(query_embeddings @ embeddings.T)
print(embeddings[0] @ embeddings[1].T)

text = []
for i in range(200):
    text.append("def f(x,y): if x<y: return y else return x "+str(i))
begin = time.time()
query_embeddings = model.encode(text, prompt_name="query")
end = time.time()
print("Time elapsed: ", end - begin)
