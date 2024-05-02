from typing import List, Literal, Union

import math

from langchain.tools.base import StructuredTool
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    create_openapi_agent,create_openai_tools_agent,
    AgentOutputParser,
)
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.prompts import StringPromptTemplate
#from langchain.llms import HuggingFaceTextGenInference
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


##########################################################
# Step 1: Define the functions you want to articulate. ###
##########################################################


def calculator(
    input_a: float,
    input_b: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
):
    """
    Computes a calculation.
    Args:
    input_a (float) : Required. The first input.
    input_b (float) : Required. The second input.
    operation (string): The operation. Choices include: add to add two numbers, subtract to subtract two numbers, multiply to multiply two numbers, and divide to divide them.
    """
    match operation:
        case "add":
            return input_a + input_b
        case "subtract":
            return input_a - input_b
        case "multiply":
            return input_a * input_b
        case "divide":
            return input_a / input_b


def cylinder_volume(radius, height):
    """
    Calculate the volume of a cylinder.
    Parameters:
    - radius (float): The radius of the base of the cylinder.
    - height (float): The height of the cylinder.
    Returns:
    - float: The volume of the cylinder.
    """
    if radius < 0 or height < 0:
        raise ValueError("Radius and height must be non-negative.")

    volume = math.pi * (radius**2) * height
    return volume


#############################################################
# Step 2: Let's define some utils for building the prompt ###
#############################################################


RAVEN_PROMPT = """
{raven_tools}
User Query: {input}<human_end>
"""



# Set up a prompt template
class RavenPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        prompt = ""
        for tool in self.tools:
            func_signature, func_docstring = tool.description.split(" - ", 1)
            prompt += f'\nFunction:\ndef {func_signature}\n"""\n{func_docstring}\n"""\n'
        kwargs["raven_tools"] = prompt
        return self.template.format(**kwargs).replace("{{", "{").replace("}}", "}")


class RavenOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Call:" in llm_output:
            return AgentFinish(
                return_values={
                    "output": llm_output.strip()
                    .replace("Call:", "")
                    .strip()
                },
                log=llm_output,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")


##################################################
# Step 3: Build the agent with these utilities ###
##################################################


inference_server_url = "http://127.0.0.1:8000/v1"


# llm = HuggingFaceTextGenInference(
#     inference_server_url=inference_server_url,
#     temperature=0.001,
#     max_new_tokens=400,
#     do_sample=False,
# )

llm = ChatOpenAI(
    base_url=inference_server_url,
    temperature=0.001,
    max_tokens=400,
   # do_sample=False,
    api_key="<YOUR OPENAI API KEY>",
)

tools = [
    StructuredTool.from_function(calculator),
    StructuredTool.from_function(cylinder_volume),
]
raven_prompt = RavenPromptTemplate(
    template=RAVEN_PROMPT, tools=tools, input_variables=["input"]
)
llm_chain = LLMChain(llm=llm, prompt=raven_prompt)
output_parser = RavenOutputParser()
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["<bot_end>"],
    allowed_tools=tools,
)



agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

call = agent_chain.invoke(
    "I have a cake that is about 3 centimenters high and 200 centimeters in radius. How much cake do I have?"
)
print(call)
#print(eval(call))
call = agent_chain.invoke("What is 1+10?")
print(call)
#print(eval(call))
