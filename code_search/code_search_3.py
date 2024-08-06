import os
from typing import List, Dict, Any, Optional
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import logging
from concurrent.futures import ThreadPoolExecutor
import ast
import re
from functools import lru_cache
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import traceback

# 加载环境变量
load_dotenv()


# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化 Elasticsearch
es = AsyncElasticsearch(
    os.getenv("ELASTICSEARCH_URL", "http://192.168.32.129:9200"), verify_certs=False
)

# 初始化 CodeBERT 模型
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化 FastAPI 应用
app = FastAPI(
    title="Code Search API",
    description="An API for indexing and searching code",
    version="1.0.0",
)


# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 线程池
executor = ThreadPoolExecutor(max_workers=4)


# Pydantic 模型
class CodeFile(BaseModel):
    file_path: str
    content: str


class BulkIndexRequest(BaseModel):
    files: List[CodeFile]


class SearchQuery(BaseModel):
    query: str
    language: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    file_path: str
    language: str
    score: float
    snippet: str


async def create_index():
    """创建 Elasticsearch 索引"""
    index_name = "code_index"
    if not await es.indices.exists(index=index_name):
        await es.indices.create(
            index=index_name,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "code_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"],
                            }
                        }
                    },
                },
                "mappings": {
                    "properties": {
                        "file_path": {"type": "keyword"},
                        "content": {"type": "text", "analyzer": "code_analyzer"},
                        "language": {"type": "keyword"},
                        "functions": {
                            "type": "nested",
                            "properties": {
                                "name": {"type": "keyword"},
                                "lineno": {"type": "integer"},
                                "code": {"type": "text", "analyzer": "code_analyzer"},
                            },
                        },
                        "vector": {"type": "dense_vector", "dims": 768},
                    }
                },
            },
        )
        logger.info(f"Index '{index_name}' created.")


def extract_functions(code: str) -> List[Dict[str, Any]]:
    """提取代码中的函数"""
    functions = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "code": ast.get_source_segment(code, node),
                    }
                )
    except SyntaxError as e:
        logger.warning(f"Failed to parse code for function extraction.: {str(e)}")
    return functions


def detect_language(code: str) -> str:
    """检测代码语言"""
    patterns = {
        "python": r"\bdef\b|\bclass\b|\bimport\b",
        "javascript": r"\bfunction\b|\bconst\b|\blet\b|\bvar\b",
        "java": r"\bclass\b|\bpublic\b|\bprivate\b|\bprotected\b",
        "c++": r"\b#include\b|\busing namespace\b|\bstd::\b",
        "ruby": r"\bdef\b|\bclass\b|\bmodule\b|\brequire\b",
    }
    for lang, pattern in patterns.items():
        if re.search(pattern, code):
            return lang
    return "unknown"


@lru_cache(maxsize=1000)
def code_to_vector(code: str) -> np.ndarray:
    """将代码转换为向量"""
    inputs = tokenizer(
        code, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


async def index_code(file_path: str, code_content: str):
    """索引代码文件"""
    try:
        language = detect_language(code_content)
        functions = extract_functions(code_content)
        vector = code_to_vector(code_content).tolist()
        doc = {
            "file_path": file_path,
            "content": code_content,
            "language": language,
            "functions": functions,
            "vector": vector,
        }
        try:
            await es.index(index="code_index", body=doc)
            logger.info(f"Indexed file: {file_path}")
        except RuntimeError as e:
            logger.error(f"Error indexing file {file_path}: {str(e)}")

    except Exception as e:
        logger.error(f"Error indexing file {file_path}: {str(e)}")


async def bulk_index_code(code_files: List[CodeFile]):
    """批量索引代码文件"""

    async def generate_actions():
        for file in code_files:
            language = detect_language(file.content)
            functions = extract_functions(file.content)
            vector = code_to_vector(file.content).tolist()
            yield {
                "_index": "code_index",
                "_source": {
                    "file_path": file.file_path,
                    "content": file.content,
                    "language": language,
                    "functions": functions,
                    "vector": vector,
                },
            }

    try:
        await async_bulk(es, generate_actions())
        logger.info(f"Bulk indexed {len(code_files)} documents")
    except Exception as e:
        logger.error(f"Error in bulk indexing: {str(e)}")


async def search_code(
    query: str, language: Optional[str] = None, top_k: int = 10
) -> List[SearchResult]:
    """搜索代码"""
    query_vector = code_to_vector(query).tolist()

    must = [{"match": {"content": query}}]
    if language:
        must.append({"term": {"language": language}})

    body = {
        "query": {
            "script_score": {
                "query": {"bool": {"must": must}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": query_vector},
                },
            }
        },
        "size": top_k,
    }

    try:
        results = await es.search(index="code_index", body=body)
        return [
            SearchResult(
                file_path=hit["_source"]["file_path"],
                language=hit["_source"]["language"],
                score=hit["_score"],
                snippet=hit["_source"]["content"][:200] + "...",
            )
            for hit in results["hits"]["hits"]
        ]
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return []


@app.post("/index")
async def index_code_api(code_file: CodeFile):
    """API endpoint for indexing code"""
    await index_code(code_file.file_path, code_file.content)
    # return {"message": "Code indexing started"}


@app.post("/bulk_index")
async def bulk_index_api(request: BulkIndexRequest, background_tasks: BackgroundTasks):
    """API endpoint for bulk indexing code"""
    background_tasks.add_task(bulk_index_code, request.files)
    return {"message": f"Bulk indexing of {len(request.files)} files started"}


@app.post("/search", response_model=List[SearchResult])
async def search_code_api(query: SearchQuery):
    """API endpoint for searching code"""
    results = await search_code(query.query, query.language, query.top_k)
    return results


@app.get("/health")
async def health_check():
    """健康检查 endpoint"""
    try:
        if await es.ping():
            return {"status": "healthy", "elasticsearch": "connected"}
        else:
            raise HTTPException(
                status_code=503, detail="Elasticsearch is not responding"
            )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


if __name__ == "__main__":
    import asyncio

    # asyncio.set_event_loop(asyncio.new_event_loop())
    asyncio.run(create_index())
    uvicorn.run(app, host="0.0.0.0", port=8090)
