
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-code",
    trust_remote_code=True
)

# control your input sequence length up to 8192
model.max_seq_length = 1024

embeddings = model.encode([
    'calculate maximum value',
    'def f(a,b): if a>b: return a else return b',
    "def f(a,b): if a<b: return a else return b"
])
print(cos_sim(embeddings[0], embeddings[1]))
print(cos_sim(embeddings[0], embeddings[2]))
print(cos_sim(embeddings[1], embeddings[2]))