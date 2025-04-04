from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class CodeIndexer:
    def __init__(self):
        self.index = {}
        self.tfidf_vectorizer = TfidfVectorizer()

    def add_to_index(self, name: str, content: str, type: str):
        self.index[name] = {
            "content": content,
            "type": type,
            "keywords": self.extract_keywords(content)
        }

    def extract_keywords(self, content: str, top_n: int = 5) -> List[str]:
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        sorted_items = sorted(zip(tfidf_matrix.tocsc().data, feature_names), key=lambda x: x[0], reverse=True)
        return [item[1] for item in sorted_items[:top_n]]

    def search(self, query: str) -> List[Dict[str, Any]]:
        query_vector = self.tfidf_vectorizer.transform([query])
        results = []
        for name, data in self.index.items():
            content_vector = self.tfidf_vectorizer.transform([data['content']])
            similarity = np.dot(query_vector.toarray(), content_vector.toarray().T)[0][0]
            results.append({"name": name, "type": data['type'], "similarity": similarity})
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:5]
