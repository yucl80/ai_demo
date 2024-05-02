sentences_1 = ["texts","sasdfas","sfasdfas","asdfsadf","asdfasfas","sdfsdfs","asdfasf","asdfasdfas","asdfsdf","asdfsdfas","asdfasfdasdf","adsfasdfasdf"]
import time
import torch
from huggingface_hub import login
login(token="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")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
_model = torch.compile(model)
model.encode(["你好，世界！"])
st = time.perf_counter()
embeddings_1 = _model.encode(sentences_1, normalize_embeddings=True)
et = time.perf_counter()
print(et - st)




from llama_cpp import Llama
model = Llama("/home/test/llm-models/bge-large-zh-v1.5-q4_k_m.gguf", embedding=True,verbose=False)
model.embed(["你好，世界！"])

st = time.perf_counter()
embed = model.embed(sentences_1)
et = time.perf_counter()
print(et - st)



# from optimum.onnxruntime import ORTModelForFeatureExtraction
# from transformers import AutoTokenizer
# import torch

# # Make sure that you download the model weights locally to `bge-m3-onnx`
# model = ORTModelForFeatureExtraction.from_pretrained("bge-m3-onnx", provider="CUDAExecutionProvider") # omit provider for CPU usage.
# tokenizer = AutoTokenizer.from_pretrained("hooman650/bge-m3-onnx-o4")

# sentences = [
#     "English: The quick brown fox jumps over the lazy dog.",
#     "Spanish: El rápido zorro marrón salta sobre el perro perezoso.",
#     "French: Le renard brun rapide saute par-dessus le chien paresseux.",
#     "German: Der schnelle braune Fuchs springt über den faulen Hund.",
#     "Italian: La volpe marrone veloce salta sopra il cane pigro.",
#     "Japanese: 速い茶色の狐が怠惰な犬を飛び越える。",
#     "Chinese (Simplified): 快速的棕色狐狸跳过懒狗。",
#     "Russian: Быстрая коричневая лиса прыгает через ленивую собаку.",
#     "Arabic: الثعلب البني السريع يقفز فوق الكلب الكسول.",
#     "Hindi: तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूद जाती है।"
# ]

# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to("cuda")

# # Get the embeddings
# out=model(**encoded_input,return_dict=True).last_hidden_state

# # normalize the embeddings
# dense_vecs = torch.nn.functional.normalize(out[:, 0], dim=-1)

