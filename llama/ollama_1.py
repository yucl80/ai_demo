from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

llm.invoke("Tell me a joke")