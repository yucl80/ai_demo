from FlagEmbedding import FlagLLMModel
queries = ["how much protein should a female eat", "summit define"]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
model = FlagLLMModel('BAAI/bge-multilingual-gemma2', 
                     query_instruction_for_retrieval="Given a web search query, retrieve relevant passages that answer the query.",
                     use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode_queries(queries)
embeddings_2 = model.encode_corpus(documents)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# [[ 0.559     0.01654 ]
# [-0.002575  0.4998  ]]
