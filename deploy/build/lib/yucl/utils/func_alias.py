from .llm_builder import create_llm
from .HttpEmbeddingsClient import HttpEmbeddingsClient

def ChatOpenAI(temperature=0.01, timeout=180, max_retries=0,streaming=False):
    return create_llm(temperature=temperature, timeout=timeout, max_retries=max_retries,streaming=streaming)

def OpenAIEmbeddings():
    return HttpEmbeddingsClient()