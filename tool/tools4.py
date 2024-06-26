#https://blog.csdn.net/yuanmintao/article/details/136268609?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field


from langchain.agents import create_structured_chat_agent,AgentExecutor
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_community.tools import ShellTool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.tools import PythonREPLTool
import json
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from ChatGLM4 import ChatZhipuAI
import time
import langchain
#langchain.debug = True
from typing import List
from yucl.utils import  create_llm


class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
    )

def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

pythonREPLTool = PythonREPLTool()


#prompt = ChatPromptTemplate.from_messages(   [        ("system", "You are a helpful assistant"),        ("user", "{input}"),        MessagesPlaceholder(variable_name="agent_scratchpad"),    ])
#prompt = hub.pull("hwchase17/structured-chat-agent")
#prompt = hub.pull("hwchase17/openai-functions-agent")
prompt = hub.pull("rlm/rag-prompt")
#prompt.pretty_print()


llm = create_llm()    
    

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)    
# embedding = hf.embed_query("hi this is harrison")
    
# Load in document to retrieve over
loader = TextLoader("/home/test/src/state_of_the_union.txt")
documents = loader.load()


# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Here is where we add in the fake source information
for i, doc in enumerate(texts):
    doc.metadata["page_chunk"] = i

vectorstore = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "state-of-union-retriever",
    "Query a retriever to get information about state of the union address",
)  
    

llm_with_tools = llm.bind_functions([retriever_tool, Response])

agent = (
    {
        "input": lambda x: x["input"],
        # Format agent scratchpad from intermediate steps
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | parse
)
# 创建 structured chat agent
#agent = create_structured_chat_agent(llm, tools, prompt)

#tools = [pythonREPLTool,  multiply,]
tools = [retriever_tool]

#agent = create_structured_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, handle_parsing_errors=True, verbose=True)
#agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

st = time.perf_counter()
rep = agent_executor.invoke(
    {"input": "what did the president say about ketanji brown jackson"},
    return_only_outputs=True,
)
#rep = agent_executor.invoke(  { "input": "2 乘 4 等于几 ？"   })
et = time.perf_counter() - st
print("invoke time:", et)
print(rep)


