from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import AIMessage
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")


template = """{question}"""
prompt = PromptTemplate.from_template(template)

messages = [
    AIMessage(content="我将从美国到中国来旅游，出行前希望了解中国的城市"),
    AIMessage(content="欢迎问我任何问题。"),
]



llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "北京和上海两座城市有什么不同？"

rep = llm_chain.invoke(question)

print(rep)