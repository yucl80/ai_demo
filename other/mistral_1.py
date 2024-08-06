from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("D:\\llm\\mistral-7b-instruct-v0.2-ONNX")
model = ORTModelForCausalLM.from_pretrained("D:\\llm\\mistral-7b-instruct-v0.2-ONNX")
inputs = tokenizer("请介绍一下中国的教育制度", return_tensors="pt")
gen_tokens = model.generate(**inputs,max_length=5000)
print(tokenizer.batch_decode(gen_tokens))

