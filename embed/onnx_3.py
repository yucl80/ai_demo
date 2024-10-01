# pip install "optimum[onnxruntime-gpu]" transformers

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('D:\\llm\\jina-embeddings-onnx')
model = ORTModelForSequenceClassification.from_pretrained('D:\\llm\\jina-embeddings-onnx')
#model.to("cpu")

pairs = [['what is panda?', 'hi'],['i love panda?', '我喜欢熊猫'], ['how are you?', '你好'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    data = model(**inputs, return_dict=True)
    print(data)


