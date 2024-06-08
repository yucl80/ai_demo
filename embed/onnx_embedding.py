import os
import transformers
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForCustomTasks
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import time

print(ort.get_available_providers())

tokenizer = AutoTokenizer.from_pretrained("/home/test/tmp/quantized_bge-m3_onnx")

model = ORTModelForCustomTasks.from_pretrained(
    "/home/test/tmp/quantized_bge-m3_onnx", provider="AzureExecutionProvider"
)

def embedding(input):
    inputs = tokenizer(
        text=input,
        padding="longest",
        return_tensors="np",
    )
    inputs = {key: value.astype(np.int64) for key, value in inputs.items()}
    # inputs_onnx = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()}
    embeds = model.forward(**inputs)["sentence_embedding"].tolist()
    return embeds

input = "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."

embeds = embedding(input)

print(embeds)

begin = time.time()
for a in range(1000):
    embeds = embedding(["World is big "+str(a)])

end = time.time()

print(end - begin)
