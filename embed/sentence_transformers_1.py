from sentence_transformers import SentenceTransformer

# 加载预训练的嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 示例功能描述和代码片段
descriptions = ["实现一个二分查找算法", "查找数组中的最大值"]
code_snippets = [
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "def find_max(arr):\n    max_value = arr[0]\n    for num in arr:\n        if num > max_value:\n            max_value = num\n    return max_value"
]

# 生成嵌入
description_embeddings = model.encode(descriptions, batch_size=8, convert_to_tensor=True)
code_embeddings = model.encode(code_snippets, batch_size=8, convert_to_tensor=True)

import torch

print(description_embeddings @ code_embeddings.T)
