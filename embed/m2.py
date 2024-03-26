# -*- coding: utf-8 -*-
"""
"""
import argparse
import uvicorn
import sys
import os
import numpy as np
import numpy.ma as ma
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
import torch
from loguru import logger
import time
from typing import List
import threading
from concurrent.futures import ThreadPoolExecutor
from text2vec import SentenceModel, cos_sim, semantic_search, Similarity, EncoderType

sys.path.append('..')
executor = ThreadPoolExecutor(max_workers=4)
pwd_path = os.path.abspath(os.path.dirname(__file__))
# Use fine-tuned model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-m3",
                    help="Model save dir or model name")
args = parser.parse_args()
s_model = SentenceModel(args.model_name_or_path)


def embeddingTask(sentence):   
    embeddings = s_model.encode(sentence)
    # 返回编码结果
    return embeddings

corpus = [
        '花呗更改绑定银行卡',
        '我什么时候开通了花呗',
        '俄罗斯警告乌克兰反对欧盟协议',
        '暴风雨掩埋了东北部；新泽西16英寸的降雪',
        '中央情报局局长访问以色列叙利亚会谈',
        '人在巴基斯坦基地的炸弹袭击中丧生',
        '如何更换花呗绑定银行卡',
        'The cat sits outside',
        'A man is playing guitar',
        'The new movie is awesome',
        '敏捷的棕色狐狸跳过了懒狗',
        '花呗更改绑定银行卡',
        'The dog plays in the garden',
        'The quick brown fox jumps over the lazy dog.',
    ]
# 1. Compute text embedding
corpus_embeddings = []
st = time.perf_counter()
#futures = [executor.submit(embeddingTask, sentence) for sentence in corpus]
#for future in futures:
#    embedding = future.result()
#    corpus_embeddings=np.append(corpus_embeddings,[embedding],axis=0)
#    corpus_embeddings.append([embedding])
corpus_embeddings = s_model.encode(corpus)
et = time.perf_counter() - st
print("init time:", et)
# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.get('/')
async def index():
    return {"message": "index, docs url: /docs"}


@app.get('/add')
async def addSentence(q: str = Query(..., min_length=1, max_length=512, title='query')):
    global corpus_embeddings
    global corpus
    try:
        st = time.perf_counter()
        n = s_model.encode(q)
        et = time.perf_counter() - st
        print("emb time :", et)
        corpus_embeddings=np.append(corpus_embeddings,[n],axis=0)
        corpus=np.append(corpus,[q],axis=0)
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400


@app.get('/search')
async def emb(q: str = Query(..., min_length=1, max_length=512, title='query')):
    try:
        st = time.perf_counter()
        query_embedding = s_model.encode(q)
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
        hits = hits[0]
        et = time.perf_counter() - st
        print("search time:", et)
        for hit in hits:
            print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
        return corpus[hits[0]['corpus_id']]
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400
    
@app.get('/search2')
async def emb(q: str = Query(..., min_length=1, max_length=512, title='query'), sentences: List[str] = Query(..., title='keys'), top_k: int = Query(5, ge=1, le=10, title='top_k')):
    try:
        futures = [executor.submit(embeddingTask, sentence) for sentence in sentences]
        query_embedding = s_model.encode(q)

        embeddings = corpus_embeddings
        for future in futures:
          embedding = future.result()         
          embeddings = np.append(embeddings,[embedding],axis=0)      
        
        hits = semantic_search(query_embedding, embeddings, top_k=top_k)
        hits = hits[0]
        for hit in hits:
            print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
        return corpus[hits[0]['corpus_id']]
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400
     

@app.get('/searchMulti')
async def emb(q: str = Query(..., min_length=1, max_length=512, title='query'), sentences: List[str] = Query(..., title='keys'), top_k: int = Query(5, ge=1, le=10, title='top_k')):
    try:
        st = time.perf_counter()

        futures = [executor.submit(embeddingTask, sentence) for sentence in sentences]
        embeddings = []
        for future in futures:
          embedding = future.result()
          embeddings.append(embedding) 
        
        query_embedding = s_model.encode(q)
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
        hits = hits[0]    

        et = time.perf_counter() - st
        print("search time:", et)
        for hit in hits:
            print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
        return corpus[hits[0]['corpus_id']]
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400

@app.get('/delete')
async def deleteSentence(q: str = Query(..., min_length=1, max_length=512, title='query')):
    global corpus_embeddings
    global corpus
    try:
        index = corpus.index(q)
        corpus.remove(q)
        corpus_embeddings = np.delete(corpus_embeddings, index, axis=0)
        return {'status': True, 'msg': 'Sentence deleted successfully'}
    except ValueError:
        return {'status': False, 'msg': 'Sentence not found'}
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400

@app.get('/list')
async def listSentences():
    return {'sentences': corpus}

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8001)

