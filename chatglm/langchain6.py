from langchain.chains import LLMSummarizationCheckerChain
from langchain_openai import ChatOpenAI
from code.deploy.yucl_utils.jwt_token import get_api_key
from ChatGLM4 import ChatZhipuAI
import os
#llm = OpenAI(temperature=0)


llm = ChatZhipuAI(
#   endpoint_url="https://open.bigmodel.cn/api/paas/v4",
   endpoint_url="http://127.0.0.1:8000/v1",
   temperature=0.1,
   api_key=get_api_key(),
   model_name="glm-4",
)
os.environ["OPENAI_API_KEY"] = "NOKEY";
llm = ChatOpenAI(temperature=0,base_url="http://127.0.0.1:8000/v1/",api_key="NOKEY")
checker_chain = LLMSummarizationCheckerChain.from_llm(llm, verbose=True, max_checks=2)
text = """
Your 9-year old might like these recent discoveries made by The James Webb Space Telescope (JWST):
• In 2023, The JWST spotted a number of galaxies nicknamed "green peas." They were given this name because they are small, round, and green, like peas.
• The telescope captured images of galaxies that are over 13 billion years old. This means that the light from these galaxies has been traveling for over 13 billion years to reach us.
• JWST took the very first pictures of a planet outside of our own solar system. These distant worlds are called "exoplanets." Exo means "from outside."
These discoveries can spark a child's imagination about the infinite wonders of the universe."""
rep = checker_chain.invoke(text)
print(rep)