from transformers import RobertaTokenizer, RobertaModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

# 加载GraphCodeBERT模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

# 将代码片段向量化
def encode_code(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach()  # 取最后一层的平均值作为表示

# 功能描述向量化
def encode_description(description):
    inputs = tokenizer(description, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach()

# 计算余弦相似度
def compute_similarity(vec1, vec2):
    return cosine_similarity(vec1, vec2)

# 从文件中加载代码片段
def load_code_snippets(directory):
    code_snippets = []
    for filename in os.listdir(directory):
        if filename.endswith(".py"):  # 只处理.py文件
            with open(os.path.join(directory, filename), "r",encoding="utf-8") as file:
                code_snippets.append(file.read())
    return code_snippets

# 将长代码片段分块
def split_code_into_chunks(code_snippet, chunk_size=512):
    lines = code_snippet.splitlines()
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        line_length = len(tokenizer.tokenize(line))
        if current_length + line_length > chunk_size:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(line)
        current_length += line_length

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

# 根据功能描述搜索代码片段
def search_code_by_description(description, code_snippets):
    description_vec = encode_description(description)
    best_match = None
    best_similarity = -1

    for snippet in code_snippets:
        code_chunks = split_code_into_chunks(snippet)  # 对每个代码片段分块
        for chunk in code_chunks:
            chunk_vec = encode_code(chunk)
            similarity = compute_similarity(description_vec, chunk_vec).item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = chunk

    return best_match

# 示例代码
if __name__ == "__main__":
    # 加载代码片段
    code_snippets = load_code_snippets("D:\\workspaces\\python_projects\\ai_demo\\embed")

    # 输入功能描述
    description = "sql query"

    # 搜索匹配的代码片段
    best_match = search_code_by_description(description, code_snippets)
    print("最匹配的代码片段:\n", best_match)
