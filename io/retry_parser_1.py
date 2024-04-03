from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
)
from langchain.prompts import (
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAI

def print_output(output) -> any:
    print("Output Beings:----")
    print(output)
    print("Output Ends----")
    return output

template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""


class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


parser = PydanticOutputParser(pydantic_object=Action)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt_value = prompt.format_prompt(query="how can i search for a movie?")

bad_response = '{"action": "search"}'


fix_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(base_url="http://localhost:8000/v1", openai_api_key="your_api_key"))
#fix_parser.parse(bad_response)

from langchain.output_parsers import RetryOutputParser
retry_parser = RetryOutputParser.from_llm(parser=parser, llm=ChatOpenAI(base_url="http://localhost:8000/v1", openai_api_key="your_api_key",temperature=0))
#retry_parser.parse_with_prompt(bad_response, prompt_value)

from langchain_core.runnables import RunnableLambda, RunnableParallel

completion_chain = prompt | ChatOpenAI(base_url="http://localhost:8000/v1", openai_api_key="your_api_key",temperature=0) |print_output

main_chain = RunnableParallel(
    completion=completion_chain, prompt_value=prompt
) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))


main_chain.invoke({"query": "how can i search for a movie?"})