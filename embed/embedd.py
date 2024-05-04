import onnxruntime as ort
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from optimum.onnxruntime import ORTModelForCustomTasks

model_path = "/home/test/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/babcf60cae0a1f438d7ade582983d4ba462303c2/onnx/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
local_onnx_model = ORTModelForCustomTasks.from_pretrained(model_path)
import torch

from llama_cpp import Llama
model = Llama("/home/test/llm-models/bge-large-zh-v1.5-q4_k_m.gguf", embedding=True,verbose=False)


def embed(text) -> any:
    inputs = tokenizer(
        text,
        padding="longest",
        return_tensors="np",
    )
    return local_onnx_model.forward(**inputs)['sentence_embedding'][0]  

corpus = [
    "我爱你",
    "我恨你",
    "我喜欢你",
    "我讨厌你",
    "我不喜欢你",
    "我不爱你",
    "I love you",
    "I like you",
    "I hate you",
    "I don't like you",
]
import time
corpus_embeddings = []
for str in corpus:
    begin_time = time.time()
    data = model.embed(str)
    end_time = time.time()
    print("Time taken for embedding: ", end_time - begin_time)
    corpus_embeddings.append(torch.tensor(data))

query_embedding = embed("我爱你")

hits = util.semantic_search(torch.tensor(query_embedding), corpus_embeddings, top_k=5)
hits = hits[0]  # Get the hits for the first query
for hit in hits:
    print(corpus[hit["corpus_id"]], "(Score: {:.4f})".format(hit["score"]))
