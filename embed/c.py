sentences_1 = ["texts","sasdfas","sfasdfas","asdfsadf","asdfasfas","sdfsdfs","asdfasf","asdfasdfas","asdfsdf","asdfsdfas","asdfasfdasdf","adsfasdfasdf"]
import time


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
model.encode(["你好，世界！"])
st = time.perf_counter()
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
et = time.perf_counter()
print(et - st)




from llama_cpp import Llama
model = Llama("/home/test/llm-models/bge-large-zh-v1.5-q4_k_m.gguf", embedding=True,verbose=False)
model.embed(["你好，世界！"])

st = time.perf_counter()
embed = model.embed(sentences_1)
et = time.perf_counter()
print(et - st)



