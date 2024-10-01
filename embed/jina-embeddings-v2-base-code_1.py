
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-code",
    trust_remote_code=True
)

# control your input sequence length up to 8192
model.max_seq_length = 1024

embeddings = model.encode([
    'How do I access the index while iterating over a sequence with a for loop?',
    '# Use the built-in enumerator\nfor idx, x in enumerate(xs):\n    print(idx, x)',
])
print(cos_sim(embeddings[0], embeddings[1]))
