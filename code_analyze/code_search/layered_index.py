from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import networkx as nx
import json
import os

class LayeredIndex:
    def __init__(self, vector_model: str = 'all-MiniLM-L6-v2'):
        # 语义层索引
        self.semantic_model = SentenceTransformer(vector_model)
        self.semantic_index = {}
        
        # 文本层索引
        self.tfidf_vectorizer = TfidfVectorizer()
        self.text_index = {}
        
        # 依赖层索引
        self.dependency_graph = nx.DiGraph()
        
        # 业务逻辑层索引
        self.business_index = {}
        
        # 元数据索引
        self.metadata = {}
        
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any], 
                    dependencies: List[str] = None, business_logic: Dict[str, Any] = None):
        """添加文档到多层索引"""
        # 1. 更新语义层索引
        semantic_vector = self.semantic_model.encode(content)
        self.semantic_index[doc_id] = semantic_vector
        
        # 2. 更新文本层索引
        if not self.text_index:
            self.tfidf_vectorizer.fit([content])
        text_vector = self.tfidf_vectorizer.transform([content]).toarray()[0]
        self.text_index[doc_id] = text_vector
        
        # 3. 更新依赖层索引
        if dependencies:
            for dep in dependencies:
                self.dependency_graph.add_edge(doc_id, dep)
        
        # 4. 更新业务逻辑层索引
        if business_logic:
            self.business_index[doc_id] = business_logic
            
        # 5. 更新元数据
        self.metadata[doc_id] = metadata
        
    def remove_document(self, doc_id: str):
        """从索引中移除文档"""
        if doc_id in self.semantic_index:
            del self.semantic_index[doc_id]
        if doc_id in self.text_index:
            del self.text_index[doc_id]
        if doc_id in self.business_index:
            del self.business_index[doc_id]
        if doc_id in self.metadata:
            del self.metadata[doc_id]
        if self.dependency_graph.has_node(doc_id):
            self.dependency_graph.remove_node(doc_id)
            
    def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any],
                       dependencies: List[str] = None, business_logic: Dict[str, Any] = None):
        """更新文档索引"""
        self.remove_document(doc_id)
        self.add_document(doc_id, content, metadata, dependencies, business_logic)
        
    def search(self, query: str, layer: str = 'all', top_k: int = 5) -> List[Dict[str, Any]]:
        """多层索引搜索"""
        results = []
        
        if layer in ['all', 'semantic']:
            # 语义层搜索
            query_vector = self.semantic_model.encode(query)
            semantic_scores = {}
            for doc_id, vector in self.semantic_index.items():
                score = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                semantic_scores[doc_id] = score
                
        if layer in ['all', 'text']:
            # 文本层搜索
            query_vector = self.tfidf_vectorizer.transform([query]).toarray()[0]
            text_scores = {}
            for doc_id, vector in self.text_index.items():
                score = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                text_scores[doc_id] = score
                
        if layer == 'all':
            # 融合搜索结果
            combined_scores = {}
            for doc_id in self.semantic_index.keys():
                combined_scores[doc_id] = (
                    0.6 * semantic_scores.get(doc_id, 0) +  # 语义层权重
                    0.4 * text_scores.get(doc_id, 0)        # 文本层权重
                )
        elif layer == 'semantic':
            combined_scores = semantic_scores
        elif layer == 'text':
            combined_scores = text_scores
            
        # 排序并返回结果
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for doc_id, score in sorted_docs:
            result = {
                'doc_id': doc_id,
                'score': score,
                'metadata': self.metadata.get(doc_id, {}),
            }
            
            # 添加依赖信息
            if self.dependency_graph.has_node(doc_id):
                result['dependencies'] = {
                    'incoming': list(self.dependency_graph.predecessors(doc_id)),
                    'outgoing': list(self.dependency_graph.successors(doc_id))
                }
                
            # 添加业务逻辑信息
            if doc_id in self.business_index:
                result['business_logic'] = self.business_index[doc_id]
                
            results.append(result)
            
        return results
    
    def get_dependencies(self, doc_id: str, depth: int = 1) -> Dict[str, List[str]]:
        """获取文档的依赖关系"""
        if not self.dependency_graph.has_node(doc_id):
            return {'incoming': [], 'outgoing': []}
            
        incoming = set()
        outgoing = set()
        
        # 获取入边依赖
        current_nodes = {doc_id}
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                predecessors = set(self.dependency_graph.predecessors(node))
                incoming.update(predecessors)
                next_nodes.update(predecessors)
            current_nodes = next_nodes
            
        # 获取出边依赖
        current_nodes = {doc_id}
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                successors = set(self.dependency_graph.successors(node))
                outgoing.update(successors)
                next_nodes.update(successors)
            current_nodes = next_nodes
            
        return {
            'incoming': list(incoming),
            'outgoing': list(outgoing)
        }
    
    def save_index(self, save_dir: str):
        """保存索引到文件"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存语义索引
        np.save(os.path.join(save_dir, 'semantic_vectors.npy'),
                {k: v.tolist() for k, v in self.semantic_index.items()})
        
        # 保存文本索引
        np.save(os.path.join(save_dir, 'text_vectors.npy'),
                {k: v.tolist() for k, v in self.text_index.items()})
        
        # 保存依赖图
        nx.write_gpickle(self.dependency_graph,
                        os.path.join(save_dir, 'dependency_graph.gpickle'))
        
        # 保存业务逻辑索引
        with open(os.path.join(save_dir, 'business_index.json'), 'w') as f:
            json.dump(self.business_index, f)
            
        # 保存元数据
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f)
            
    def load_index(self, save_dir: str):
        """从文件加载索引"""
        # 加载语义索引
        semantic_vectors = np.load(os.path.join(save_dir, 'semantic_vectors.npy'),
                                 allow_pickle=True).item()
        self.semantic_index = {k: np.array(v) for k, v in semantic_vectors.items()}
        
        # 加载文本索引
        text_vectors = np.load(os.path.join(save_dir, 'text_vectors.npy'),
                             allow_pickle=True).item()
        self.text_index = {k: np.array(v) for k, v in text_vectors.items()}
        
        # 加载依赖图
        self.dependency_graph = nx.read_gpickle(
            os.path.join(save_dir, 'dependency_graph.gpickle'))
        
        # 加载业务逻辑索引
        with open(os.path.join(save_dir, 'business_index.json'), 'r') as f:
            self.business_index = json.load(f)
            
        # 加载元数据
        with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
