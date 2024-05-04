from optimum.onnxruntime import ORTModelForSequenceClassification  # type: ignore

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
model_ort = ORTModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base', file_name="onnx/model.onnx")



# Sentences we want sentence embeddings for
pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

# Tokenize sentences
encoded_input = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')

scores_ort = model_ort(**encoded_input, return_dict=True).logits.view(-1, ).float()
# Compute token embeddings
with torch.inference_mode():
    scores = model_ort(**encoded_input, return_dict=True).logits.view(-1, ).float()

# scores and scores_ort are identica