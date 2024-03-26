from llama_cpp import Llama
model = Llama("/home/test/.cache/ggufs/bge-large-zh-v1.5-q4_k_m.gguf", embedding=True)
import time

st = time.perf_counter()
embed = model.embed("texts")
et = time.perf_counter()
print(et - st)

