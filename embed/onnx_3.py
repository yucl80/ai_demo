# pip install "optimum[onnxruntime-gpu]" transformers

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('swulling/bge-reranker-large-onnx-o4')
model = ORTModelForSequenceClassification.from_pretrained('swulling/bge-reranker-large-onnx-o4')
#model.to("cpu")

pairs = [['what is panda?', 'hi'],['i love panda?', '我喜欢熊猫'], ['how are you?', '你好'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
