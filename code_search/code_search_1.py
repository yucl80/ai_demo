import os
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify

# 初始化 Elasticsearch
es = Elasticsearch([{'scheme': 'http','host': '192.168.32.129', 'port': 9200}])

# 初始化 BERT 模型
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# 初始化 Flask 应用
app = Flask(__name__)

def index_code(file_path, code_content):
    """索引代码文件"""
    doc = {
        'file_path': file_path,
        'content': code_content
    }
    es.index(index="code_index", body=doc)

def code_to_vector(code):
    """将代码转换为向量"""
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def search_code(query):
    """搜索代码"""
    # 将查询转换为向量
    query_vector = code_to_vector(query)
    
    # 从 Elasticsearch 获取所有代码
    results = es.search(index="code_index", body={"query": {"match_all": {}}}, size=1000)
    
    # 计算相似度并排序
    similarities = []
    for hit in results['hits']['hits']:
        code_vector = code_to_vector(hit['_source']['content'])
        similarity = cosine_similarity([query_vector], [code_vector])[0][0]
        similarities.append((hit['_source']['file_path'], similarity))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:10]  # 返回前10个结果

@app.route('/index', methods=['POST'])
def index_code_api():
    """API endpoint for indexing code"""
    file_path = request.json['file_path']
    code_content = request.json['content']
    index_code(file_path, code_content)
    return jsonify({"message": "Code indexed successfully"}), 200

@app.route('/search', methods=['GET'])
def search_code_api():
    """API endpoint for searching code"""
    query = request.args.get('q')
    results = search_code(query)
    return jsonify(results), 200

if __name__ == '__main__':
    # 示例：索引一些代码
    sample_code = """
    def hello_world():
        print("Hello, World!")
    """
    index_code("sample.py", sample_code)
    
    # 运行 Flask 应用
    app.run(debug=True)