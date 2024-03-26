from langchain_community.retrievers import EmbedchainRetriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
# create a retriever with default options
retriever = EmbedchainRetriever.create()

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)