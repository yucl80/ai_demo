
 
 
 
 # 持久化数据库并且关闭匿名化遥测数据
 client_settings = chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True)
 # 向量模型设置cpu模式
 emb_fun = HuggingFaceEmbeddings(model_name= "GanymedeNil/text2vec-large-chinese", 
           model_kwargs={'device': "cpu"})
  """
  collection_name : 集合名称
  embedding_function: 向量模型,可以带着地址
  persist_directory 数据库地址
  client_settings 链接配置: 这里持久化数据库并且关闭匿名化遥测数据
  collection_metadata 集合配置: 这里指定了相似度匹配算法
  """
  db = Chroma(collection_name=collection_name, embedding_function=emb_fun,
                    persist_directory=persist_directory, client_settings=client_settings,
                    collection_metadata={"hnsw:space": "cosine"})
   # 查询归一化的距离
    r = db.similarity_search_with_relevance_scores(query=question, k=k, filter=_filter)
   # 相似度评分
   score = r[1]