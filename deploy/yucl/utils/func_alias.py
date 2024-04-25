from .llm_builder import create_llm
from .HttpEmbeddingsClient import HttpEmbeddingsClient

def ChatOpenAI(model="glm-3-turbo",temperature=0.01, timeout=180, max_retries=0,streaming=False):
    return create_llm(model = model, temperature=temperature, timeout=timeout, max_retries=max_retries,streaming=streaming)

def OpenAIEmbeddings(model="bge-large-zh-v1.5"):
    return HttpEmbeddingsClient(model=model)