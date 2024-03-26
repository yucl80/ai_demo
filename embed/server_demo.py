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
from sentence_transformers import SentenceTransformer
import numpy as np
from annoy import AnnoyIndex
import threading

lock = threading.Lock()
sys.path.append('..')
from text2vec import SentenceModel, cos_sim, semantic_search, Similarity, EncoderType

pwd_path = os.path.abspath(os.path.dirname(__file__))
# Use fine-tuned model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="shibing624/text2vec-base-multilingual",
                    help="Model save dir or model name")
args = parser.parse_args()
s_model = SentenceModel(args.model_name_or_path)

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
        'A woman watches TV',
        'The new movie is so great',
        'The quick brown fox jumps over the lazy dog.',
    ]
# 1. Compute text embedding
corpus_embeddings = s_model.encode(corpus)

annoy_index = AnnoyIndex(384, 'angular')

for i in range(len(corpus_embeddings)):
    annoy_index.add_item(i, corpus_embeddings[i])

annoy_index.build(10)

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
        if q in corpus:
            return {'status': False, 'msg': '句子已经存在。'}
                
        n = s_model.encode(q)
        lock.acquire()
        
        corpus_embeddings=np.append(corpus_embeddings,[n],axis=0)
        corpus=np.append(corpus,[q],axis=0)
        
        new_index = len(corpus_embeddings) - 1
        annoy_index.add_item(new_index, corpus_embeddings[new_index])
        annoy_index.build(10)
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400
    finally:
        lock.release()
        
@app.get('/search')
async def search(q: str = Query(..., min_length=1, max_length=512, title='query')):
    try:
        query_embedding = s_model.encode(q)       
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
        hits = hits[0]
        for hit in hits:
          print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
        return corpus[hits[0]['corpus_id']] 
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400

@app.get('/search2')
async def search2(q: str = Query(..., min_length=1, max_length=512, title='query')):
    try:
        query_embedding = s_model.encode(q)    
        hits,scores = annoy_index.get_nns_by_vector(query_embedding, 5, include_distances=True)
        similar_sentences = [corpus[i] for i in hits]
        for sentence in similar_sentences:
            print(sentence)    
        return similar_sentences; 
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400

@app.get('/delete')
async def deleteSentence(q: str = Query(..., min_length=1, max_length=512, title='query')):
    global corpus_embeddings
    global corpus
    global annoy_index
    try:
        if q not in corpus:
            return {'status': False, 'msg': '句子不存在。'}
        lock.acquire()
        index = corpus.index(q)
        corpus.remove(q)
        corpus_embeddings = np.delete(corpus_embeddings, index, axis=0)
        
        annoy_index.unload()
        annoy_index = AnnoyIndex(384, 'angular')  # Create a new Annoy index
        for i in range(len(corpus_embeddings)):
            annoy_index.add_item(i, corpus_embeddings[i])
        annoy_index.build(10)  # Build the new index
        
        return {'status': True, 'msg': 'Sentence deleted successfully'}
    except ValueError:
        return {'status': False, 'msg': 'Sentence not found'}
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400
    finally:
        lock.release()

@app.get('/list')
async def listSentences():
    return {'sentences': corpus}

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8001)
