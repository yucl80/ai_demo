#https://blog.csdn.net/yuanmintao/article/details/136146210
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import AIMessage
from yucl.utils import get_api_key
from ChatGLM4 import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


template = """{question}"""
prompt = PromptTemplate.from_template(template)

endpoint_url = "https://open.bigmodel.cn/api/paas/v4"


llm = ChatZhipuAI(
   endpoint_url=endpoint_url,
   temperature=0.1,
   api_key=get_api_key(),
   model_name="glm-4",
)

# 写作改进

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名世界级的新闻学专业的教授。"),
    ("user", "{input}")
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
rep = chain.invoke({"input": "请你作为一名新闻学专业的教授，帮助你的学生润色论文，以下是需要润色的内容：\n随着青年世代逐渐成熟， Z世代正逐步登上社会发展的大舞台，成为职场的新生主力军。进入职场的第一步就是进行职业选择，对于 Z 世代来说，职业选择可能面临着许多不同的挑战和机遇 。一方面是，相比起前几代人，Z世代成长于鼓励创新和自由民主的社会大环境下，他们有着更加开放和自由的思维方式，更加注重自我实现和价值观的匹配；另一方面是他们深受数字化的影响，在互联网和社交媒体的熏陶下，他们对多元文化有更深的认同感。因此，他们在进行职业选择时所关注的维度也更加多元化，同时也更加关注自身，进行职业选择的标尺也更为宽幅、更为多变，择业不再仅受单一因素左右，而是从多种角度考虑。?"})


print(rep)

#信息提取
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名世界级的企业架构师。"),
    ("user", "{input}")
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
rep = chain.invoke({"input": "2022年11月4日，计算机系通过线上线下相结合的方式在东主楼10-103会议室召开博士研究生导师交流会。计算机学科学位分委员会主席吴空，计算机系副主任张建、党委副书记李伟出席会议，博士生研究生导师和教学办工作人员等30余人参加会议，会议由张建主持。\n 请你提取包含“人""(name, position)，“时间”，“事件""，“地点”（location）类型的所有信息，并输出JSON格式"})


print(rep)

#rep = chain.invoke({"input": "请解读这篇文章： https://arxiv.org/abs/2303.17568"})


#print(rep)