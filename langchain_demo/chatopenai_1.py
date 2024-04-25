from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import time

llm = ChatOpenAI(model="chatglm",  api_key="your_api_key",
                 base_url="http://localhost:8000/v1", temperature=0.01)

messages = [
    ("system", "You are a helpful assistant that translates English to French."),
    ("human", "Translate this sentence from English to French. I love programming."),
]
# llm.invoke(messages)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
# chain.invoke(
#     {
#         "input_language": "English",
#         "output_language": "German",
#         "input": "I love programming.",
#     }
# )


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(...,
                          description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])

begin_time = time.time()

ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)

end_time = time.time()
print(f"Time taken: {end_time - begin_time:.2f} seconds")

print(ai_msg)
