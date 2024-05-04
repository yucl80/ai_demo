from rank_bm25 import BM25Okapi

from FlagEmbedding import FlagReranker
# Setting use_fp16 to True speeds up computation with a slight performance degradation
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)

# score = reranker.compute_score(['query', 'passage'])
# print(score)


scores = reranker.compute_score([['I love you', '我爱你'], ['I love you', 'i like you'], [
                                'I love you', '我喜欢你'], ['I love you', 'i hate you'], ['I love you', '我不爱你'], ['I love you', 'I don\'t like you']])
print(scores)


# corpus = [
#     '我爱你', 'I like you', '我喜欢你', 'I hate you', '我不爱你', 'I don\'t like you'
# ]

# bm25 = BM25Okapi(corpus)
# query = "I love you"
# r = bm25.get_top_n(query, corpus, n=2)
# print(r)
