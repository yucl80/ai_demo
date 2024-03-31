import openai
import time

client = openai.OpenAI(
    base_url="http://localhost:8000/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)

st = time.perf_counter()

completion = client.chat.completions.create(
model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": "你是一名秘书，本次负责编写会议纪要"},
    {"role": "user", "content": "2022年11月4日，计算机系通过线上线下相结合的方式在东主楼10-103会议室召开博士研究生导师交流会。计算机学科学位分委员会主席吴空，计算机系副主任张建、党委副书记李伟出席会议，博士生研究生导师和教学办工作人员等30余人参加会议，会议由张建主持。\n 请你提取包含“人""(name, position)，“时间”，“事件""，“地点”（location）类型的所有信息，并输出JSON格式"}
]
)

et = time.perf_counter()

print("used time:" + str(et-st))

print(completion.choices[0].message)