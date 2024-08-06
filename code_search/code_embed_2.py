from transformers import AutoModel, AutoTokenizer
import torch

# 选择使用的预训练模型和tokenizer
# model_name = "huggingface/CodeBERTa-small-v1"
model_name = "neulab/codebert-java"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 示例代码文本
code_text = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
"""

# 使用tokenizer对代码文本进行tokenize和编码
inputs = tokenizer(code_text, return_tensors="pt")

# 将编码后的输入传入模型获取嵌入表示
with torch.no_grad():
    outputs = model(**inputs)

# 获取最后一层的CLS token作为整体代码的嵌入表示
embeddings = outputs.last_hidden_state[:, 0, :]

# 打印嵌入表示的维度
print("嵌入表示的维度：", embeddings.size())

# 可以在这里将嵌入表示用于后续的任务，比如相似度计算、分类等
