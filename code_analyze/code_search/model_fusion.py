from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class ModelFusion:
    def __init__(self, models_config: List[Dict[str, Any]]):
        """
        初始化多模型融合系统
        
        models_config格式示例：
        [
            {
                'name': 'sentence-bert',
                'type': 'sentence_transformers',
                'model_name': 'all-MiniLM-L6-v2',
                'weight': 0.4
            },
            {
                'name': 'code-bert',
                'type': 'huggingface',
                'model_name': 'microsoft/codebert-base',
                'weight': 0.4
            },
            {
                'name': 'tfidf',
                'type': 'tfidf',
                'weight': 0.2
            }
        ]
        """
        self.models = {}
        self.weights = {}
        self.vectorizers = {}
        
        for config in models_config:
            model_name = config['name']
            model_type = config['type']
            
            if model_type == 'sentence_transformers':
                self.models[model_name] = SentenceTransformer(config['model_name'])
            elif model_type == 'huggingface':
                self.models[model_name] = {
                    'tokenizer': AutoTokenizer.from_pretrained(config['model_name']),
                    'model': AutoModel.from_pretrained(config['model_name'])
                }
            elif model_type == 'tfidf':
                self.vectorizers[model_name] = TfidfVectorizer()
                
            self.weights[model_name] = config['weight']
            
    def encode_text(self, text: str, model_name: str) -> np.ndarray:
        """使用指定模型编码文本"""
        model = self.models.get(model_name)
        
        if model is None:
            return None
            
        if isinstance(model, SentenceTransformer):
            return model.encode(text)
        elif isinstance(model, dict):  # Hugging Face models
            tokenizer = model['tokenizer']
            model = model['model']
            
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                
            # 使用最后一层隐藏状态的平均值作为文本表示
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.numpy()
            
    def get_tfidf_vector(self, text: str, vectorizer_name: str) -> np.ndarray:
        """获取TF-IDF向量"""
        vectorizer = self.vectorizers.get(vectorizer_name)
        if vectorizer is None:
            return None
            
        if not hasattr(vectorizer, 'vocabulary_'):
            vectorizer.fit([text])
            
        return vectorizer.transform([text]).toarray()[0]
        
    def compute_similarity(self, query: str, document: str) -> float:
        """计算查询和文档的相似度"""
        similarities = {}
        
        # 对每个模型计算相似度
        for model_name, weight in self.weights.items():
            if model_name in self.models:
                # 使用神经网络模型
                query_vector = self.encode_text(query, model_name)
                doc_vector = self.encode_text(document, model_name)
                
                if query_vector is not None and doc_vector is not None:
                    similarity = F.cosine_similarity(
                        torch.tensor(query_vector),
                        torch.tensor(doc_vector),
                        dim=-1
                    ).item()
                    similarities[model_name] = similarity
            else:
                # 使用TF-IDF
                query_vector = self.get_tfidf_vector(query, model_name)
                doc_vector = self.get_tfidf_vector(document, model_name)
                
                if query_vector is not None and doc_vector is not None:
                    similarity = np.dot(query_vector, doc_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                    )
                    similarities[model_name] = similarity
                    
        # 加权融合相似度分数
        final_similarity = sum(
            similarities.get(model_name, 0) * weight
            for model_name, weight in self.weights.items()
        )
        
        return final_similarity
        
    def update_weights(self, feedback_data: List[Dict[str, Any]]):
        """根据反馈更新模型权重"""
        # feedback_data格式：
        # [
        #     {
        #         'query': '查询文本',
        #         'document': '文档文本',
        #         'relevance': 1.0  # 相关性分数，范围[0, 1]
        #     },
        #     ...
        # ]
        
        # 计算每个模型的性能分数
        model_scores = {model_name: 0.0 for model_name in self.weights}
        
        for feedback in feedback_data:
            query = feedback['query']
            document = feedback['document']
            relevance = feedback['relevance']
            
            # 计算每个模型的预测分数
            for model_name in self.weights:
                if model_name in self.models:
                    query_vector = self.encode_text(query, model_name)
                    doc_vector = self.encode_text(document, model_name)
                else:
                    query_vector = self.get_tfidf_vector(query, model_name)
                    doc_vector = self.get_tfidf_vector(document, model_name)
                    
                if query_vector is not None and doc_vector is not None:
                    predicted_score = np.dot(query_vector, doc_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                    )
                    # 计算预测分数与实际相关性的差异
                    error = abs(predicted_score - relevance)
                    model_scores[model_name] += (1 - error)
                    
        # 归一化分数
        total_score = sum(model_scores.values())
        if total_score > 0:
            for model_name in self.weights:
                self.weights[model_name] = model_scores[model_name] / total_score
                
    def save_weights(self, save_path: str):
        """保存模型权重"""
        import json
        with open(save_path, 'w') as f:
            json.dump(self.weights, f)
            
    def load_weights(self, save_path: str):
        """加载模型权重"""
        import json
        with open(save_path, 'r') as f:
            self.weights = json.load(f)
