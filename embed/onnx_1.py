import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/test/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/babcf60cae0a1f438d7ade582983d4ba462303c2/onnx")
ort_session = ort.InferenceSession(
    "/home/test/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/babcf60cae0a1f438d7ade582983d4ba462303c2/onnx/model.onnx"
)
import time
start_time = time.time()
count = 200
for i in range(1,count):
    inputs = tokenizer(
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        padding="longest",
        return_tensors="np",
    )
    inputs_onnx = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()}
    outputs = ort_session.run(None, inputs_onnx)

end_time = time.time()
print("Time taken: ",( end_time - start_time)/count)
print(outputs)

from optimum.onnxruntime import ORTModelForCustomTasks

local_onnx_model = ORTModelForCustomTasks.from_pretrained(
    "/home/test/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/babcf60cae0a1f438d7ade582983d4ba462303c2/onnx/"
)



start_time = time.time()
for i in range(1,count):
    inputs = tokenizer(
        ["BGE M3 is an embedding model supporting dense retrieval","lexical matching and multi-vector interaction."],
        padding="longest",
        return_tensors="np",
    )
    outputs = local_onnx_model.forward(**inputs)
end_time = time.time()
print("Time taken: ", (end_time - start_time)/count)
print(outputs)
