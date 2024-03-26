from langchain_community.llms import ChatGLM 
llm = ChatGLM(endpoint_url="http://127.0.0.1:8000/v1/")
print(llm.invoke("你好"))
