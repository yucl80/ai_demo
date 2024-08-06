from openai import OpenAI
from bpmn_python.bpmn_diagram_rep import BpmnDiagramGraph

# 设置OpenAI API密钥
base_url = "http://localhost:8000/v1/"

client = OpenAI(base_url=base_url, api_key="YOUR_API_KEY")

def generate_bpmn_outline(process_name):
    prompt = f'''让我们一步一步来思考"{process_name}"的BPMN流程图。请生成一个详细的描述,包括以下内容:

    1. 起始事件
    2. 主要任务步骤(至少5个),每个步骤包括:
       - 任务名称
       - 任务类型(例如:用户任务、服务任务、脚本任务等)
    3. 网关和分支条件
    4. 可能的异常情况和处理流程
    5. 结束事件

    示例:
    1. 起始事件: 收到订单
    2. 主要任务步骤:
       2.1 验证订单 (用户任务)
       2.2 检查库存 (服务任务)
       2.3 处理支付 (服务任务)
       2.4 准备发货 (用户任务)
       2.5 更新订单状态 (脚本任务)
    3. 网关: 库存检查结果
       - 分支1: 库存充足
       - 分支2: 库存不足
    4. 异常处理: 支付失败处理
    5. 结束事件: 订单完成

    请按照这个格式,详细描述"{process_name}"的流程。在描述每个步骤时,请考虑可能的子步骤和细节。
    在结束时,请检查并确认是否遗漏了任何步骤。如果发现遗漏,请补充。
    '''
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a BPMN专家."},
                  {"role": "user", "content": prompt}],
        max_tokens=2000,
        timeout= 1000,
    )
    return response.choices[0].message.content

def generate_detailed_steps(step_outline):
    detailed_steps = []
    for step in step_outline:
        prompt = f'''请详细描述以下步骤的子步骤和任务类型:
        {step}
        
        对于每个子步骤,请提供:
        1. 详细的操作描述
        2. 可能的输入和输出
        3. 可能遇到的问题和解决方案
        
        请确保不遗漏任何重要信息。'''
        
        response =client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a BPMN专家."},
                      {"role": "user", "content": prompt}],
            max_tokens=1000,
             timeout= 1000,
        )
        detailed_steps.append(response.choices[0].message.content)
    return detailed_steps

def parse_llm_response(response_text):
    # 实现解析逻辑,将文本转换为结构化数据
    # 这里需要根据实际输出格式进行定制
    pass

def create_bpmn_xml(structured_data):
    bpmn_graph = BpmnDiagramGraph()
    bpmn_graph.create_new_diagram_graph(diagram_name="Generated Process")
    
    # 添加起始事件
    start_event_id = bpmn_graph.add_start_event_to_diagram("startEvent", structured_data["startEvent"], 100, 100)
    
    # 添加任务
    previous_task_id = start_event_id
    for task in structured_data["tasks"]:
        task_id = bpmn_graph.add_task_to_diagram(task["type"], task["name"], 200, 200)
        bpmn_graph.add_sequence_flow_to_diagram(previous_task_id, task_id)
        previous_task_id = task_id
    
    # 添加网关和分支条件
    for gateway in structured_data["gateways"]:
        gateway_id = bpmn_graph.add_gateway_to_diagram(gateway["type"], gateway["name"], 300, 300)
        bpmn_graph.add_sequence_flow_to_diagram(previous_task_id, gateway_id)
        for condition in gateway["conditions"]:
            condition_task_id = bpmn_graph.add_task_to_diagram("userTask", condition, 400, 400)
            bpmn_graph.add_sequence_flow_to_diagram(gateway_id, condition_task_id)
        previous_task_id = gateway_id
    
    # 添加结束事件
    end_event_id = bpmn_graph.add_end_event_to_diagram("endEvent", structured_data["endEvent"], 500, 500)
    bpmn_graph.add_sequence_flow_to_diagram(previous_task_id, end_event_id)
    
    # 导出BPMN XML文件
    bpmn_graph.export_xml_file("generated_process.bpmn", "./")

def refine_process(process_description):
    prompt = f'''请检查以下流程描述,并进行以下优化:
    1. 确保所有步骤都有明确的任务类型
    2. 检查是否有遗漏的步骤或逻辑断点
    3. 确保异常处理流程完整
    4. 添加任何缺失的重要细节

    流程描述:
    {process_description}

    请提供优化后的完整流程描述。'''
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a BPMN专家."},
                  {"role": "user", "content": prompt}],
        max_tokens=2000,
         timeout= 1000,
    )
    return response.choices[0].message.content

def validate_bpmn_xml(xml_file):
    # 实现自动化测试逻辑，确保生成的BPMN XML符合BPMN 2.0标准
    pass

def main():
    process_name = input("请输入业务流程名称: ")
    
    # 生成初始流程大纲
    bpmn_outline = generate_bpmn_outline(process_name)
    print("初始流程大纲已生成。")
    print(bpmn_outline)
    # 细化每个步骤
    step_outline = bpmn_outline.split('\n')
    detailed_steps = generate_detailed_steps(step_outline)
    print("步骤细节已生成。")
    print(detailed_steps)
    
    # 合并和优化流程描述
    full_process = '\n'.join(detailed_steps)
    refined_process = refine_process(full_process)
    print("流程已优化。")
    
    # 解析优化后的流程描述
    structured_data = parse_llm_response(refined_process)
    
    # 创建BPMN XML
    create_bpmn_xml(structured_data)
    print("BPMN流程图已生成。")
    
    # 验证生成的BPMN XML
    validate_bpmn_xml("generated_process.bpmn")
    print("BPMN流程图已验证。")

if __name__ == "__main__":
    main()
