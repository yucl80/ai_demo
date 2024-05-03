from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

loader = WebBaseLoader("https://ollama.com/blog/run-llama2-uncensored-locally")
docs = loader.load()

#llm = Ollama(model="llama2")

llm = ChatOpenAI(model="llama-3-8b",base_url="http://localhost:8000/v1", api_key="YOUR_API_KEY", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])) 

chain = load_summarize_chain(llm, chain_type="stuff")

result = chain.run(docs)
print(result)