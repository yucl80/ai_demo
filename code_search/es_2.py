import asyncio
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError

# 创建异步 Elasticsearch 客户端实例
es = AsyncElasticsearch("http://192.168.32.129:9200")

# 定义异步函数来创建索引
async def create_index(index_name, mappings):
    exists = await es.indices.exists(index_name)
    if not exists:
        await es.indices.create(index=index_name, mappings=mappings)
        print(f"Index '{index_name}' created with mappings.")

# 定义异步函数来索引文档
async def index_document(index_name, doc_id, doc_body):
    try:
        response = await es.index(index=index_name, id=doc_id, body=doc_body)
        print("Document indexed:", response['result'])
    except RequestError as e:
        print(f"Error indexing document: {e.info}")

# 定义异步函数来获取文档
async def get_document(index_name, doc_id):
    try:
        response = await es.get(index=index_name, id=doc_id)
        print("Document retrieved:", response['_source'])
    except NotFoundError:
        print(f"Document with ID {doc_id} not found in index {index_name}")

# 定义索引名称和文档ID
index_name = "my_index"
doc_id = "1"

# 定义文档数据
doc_body = {
    "name": "John Doe",
    "email": "john@example.com",
    "job": "Engineer",
}

# 定义索引的映射
mappings = {
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "email": {"type": "keyword"},
            "job": {"type": "keyword"},
        }
    }
}

# 运行异步任务
async def main():
    await create_index(index_name, mappings)
    await index_document(index_name, doc_id, doc_body)
    await get_document(index_name, doc_id)

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())