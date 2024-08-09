import json

# Define function implementations
def get_weather(location: str) -> str:
    print(f"call function get_weather")
    return f"The weather in {location} is sunny."

def calculate(expression: str) -> str:
    try:
        print(f"call function calculate")
        result = eval(expression)
        return f"The result of the calculation is: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def component_dependency_query(component_name: str) -> str:
    print(f"call function component_dependency_query")
    # Here you should add the actual code to execute the graph database query
    # Currently, we just return a simulated result
    return f"component_dependency_query: Executed query {component_name}"



# Define available functions with detailed descriptions
function_definitions = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a specified location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. 2 + 2"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "component_dependency_query",
        "description": "查询给定组件的依赖的组件和依赖该组件的其他组件",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "组件名称或组件标识"
                }
            },
            "required": ["component_name"]
        }
    }
]

function_map = {
    "get_weather": get_weather,
    "calculate": calculate,
    "component_dependency_query": component_dependency_query,
}

def get_function_definitions():
    return function_definitions

def get_functions():
    return function_map

def get_function_names():
    return ', '.join(function_map.keys())

def validate_function_call(function_name: str, function_args: dict) -> dict:
    # 判断 function_map 是否包含 function_name
    if function_name in function_map:
        # 获取对应的参数定义
        parameters = next((f["parameters"] for f in function_definitions if f["name"] == function_name), None)
        if parameters:
            required = parameters.get("required", [])
            errors = []
            # 判断function_args是否符合json规范
            try:
                args = json.loads(function_args)
                if isinstance(args,dict):
                    # 验证必需参数
                    for param in required:
                        if param not in args:
                            errors.append(f"Missing required parameter: {param}")

                    # 验证参数类型
                    for param, value in args.items():
                        if param in parameters["properties"]:
                            expected_type = parameters["properties"][param]["type"]
                            if expected_type == "string" and not isinstance(value, str):
                                errors.append(f"Parameter {param} should be of type string")
                            # 可以根据需要添加其他类型验证
                else:
                    errors.append(f"Parameter should be of type json object")
            except ValueError:
                errors.append(f"Parameter should be of type json object")

            if errors:
                return {"result": False, "errors": errors}
            return {"result": True}
    else:
        return {"result": False, "errors": f"{function_name} does not exist"}