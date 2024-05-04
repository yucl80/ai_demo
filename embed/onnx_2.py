import onnxruntime as ort
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCustomTasks
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

tokenizer = AutoTokenizer.from_pretrained("swulling/bge-reranker-large-onnx-o4")
# model_ort = ORTModelForSequenceClassification.from_pretrained(
#     "BAAI/bge-reranker-base", file_name="model.onnx"
# )

# ort_session = ort.InferenceSession("swulling/bge-reranker-large-onnx-o4/model.onnx")


# start_time = time.time()
# inputs = tokenizer(
#     "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
#     padding="longest",
#     return_tensors="np",
# )

# inputs_onnx = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()}
# outputs = ort_session.run(None, inputs_onnx)

# end_time = time.time()
# print("Time taken: ", end_time - start_time)
# print(outputs)


local_onnx_model = ORTModelForCustomTasks.from_pretrained(
    "swulling/bge-reranker-large-onnx-o4"
)

start_time = time.time()
inputs = tokenizer(
    [
        [
            "BGE M3 is an embedding model supporting dense retrieval",
            "lexical matching and multi-vector interaction.",
        ],
    ],
    padding="longest",
    return_tensors="np",
)
outputs = local_onnx_model.forward(**inputs)
end_time = time.time()
print("Time taken: ", end_time - start_time)
print(type(outputs))
print(outputs)
