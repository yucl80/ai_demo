import torch_directml
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForQuestionAnswering
from optimum.pipelines import pipeline

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

# Loading the PyTorch checkpoint and converting to the ONNX format by providing
# export=True
model = ORTModelForQuestionAnswering.from_pretrained(
    "deepset/roberta-base-squad2",
    export=True
)

onnx_qa = pipeline("question-answering", model=model, tokenizer=tokenizer, accelerator="ort")
question = "What's my name?"
context = "My name is Philipp and I live in Nuremberg."

pred = onnx_qa(question=question, context=context)
print(pred)