from zhipuai import ZhipuAI
import time
client = ZhipuAI(api_key="APIKEY") # 填写您自己的APIKey

response1 = client.chat.asyncCompletions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=[
        {
            "role": "user",
            "content": "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"
        }
    ],   
)
task_id = response1.id
task_status = ''
get_cnt = 0
 
while task_status != 'SUCCESS' and task_status != 'FAILED' and get_cnt <= 40:
    result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
    print(result_response)
    task_status = result_response.task_status
 
    time.sleep(2)
    get_cnt += 1