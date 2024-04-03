from jwt_token import get_api_key, get_api_token
from langchain_openai import ChatOpenAI

def create_llm(temperature=0.01, timeout=180):
    """create a language model using the OpenAI API."""
    return ChatOpenAI(
        model_name="glm-3-turbo",
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key= "EMPTY_KEY",
        streaming= False,
        temperature= temperature,
        timeout= timeout,
    ) 
   

def create_glm4(temperature=0.01, timeout=180):
    """create a glm-4 language model using the OpenAI API."""
    return ChatOpenAI(
        model_name="glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",       
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 180,
    ) 
    