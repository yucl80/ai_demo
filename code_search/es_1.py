from elasticsearch import Elasticsearch

# 连接到 Elasticsearch
es = Elasticsearch("http://192.168.32.129:9200")

# 定义索引名称和文档ID
index_name = "my_index"
doc_id = "1"

# 定义文档数据
doc = {
    "name": "John Doe",
    "email": "john@example.com",
    "job": "Engineer",
}

# 索引文档
response = es.index(index=index_name, id=doc_id, body=doc)
print("Index response:", response)

# 查询文档
response = es.get(index=index_name, id=doc_id)
print("Get document:", response)

# 定义带映射的索引（如果索引不存在，将创建索引及其映射）
mapping = {
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "email": {"type": "text"},
            "job": {"type": "keyword"},
        }
    }
}

# 创建索引（如果不存在）
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, mappings=mapping)
    print(f"Index '{index_name}' created with mappings.")

# 索引文档到带有映射的索引
response = es.index(index=index_name, body=doc)
print("Index response with mapping:", response)