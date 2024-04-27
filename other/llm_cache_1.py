from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
import time

# To make the caching really obvious, lets use a slower model.
llm = ChatOpenAI(model="firefunction",  api_key="your_api_key" ,base_url="http://localhost:8000/v1")

from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

begin = time.time()
# The first time, it is not yet in cache, so it should take longer
response = llm.invoke("请给我讲个笑话")
end = time.time()
print(f"First time: {end - begin:.2f}s")
print(response)
begin = time.time()
# The second time, it should be faster since it is in cache
response = llm.invoke("请给我讲个笑话")
end = time.time()
print(f"Second time: {end - begin:.2f}s")
print(response)


