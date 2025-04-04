

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
# from langchain.document_loaders.generic import GenericLoader
# from langchain.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser

repo_path = "D:\\workspaces\\python_projects\\ai_demo\\embed"

loader = GenericLoader.from_filesystem(
    repo_path, glob= "**/*",suffixes=[".py"],parser = LanguageParser(language=Language.PYTHON,parser_threshold=0)
)

documents = loader.load()

for doc in documents:
    print(doc.metadata)

python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2000,chunk_overlap=200)
texts = python_splitter.split_documents(documents)
print(len(texts))

import os
os.environ["OPENAI_API_KEY"]="NOKEY" 
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import TextEmbedEmbeddings
from HttpEmbeddingsClient import HttpEmbeddingsClient

embedding = HttpEmbeddingsClient(base_url="http://localhost:8000/v1",model="text-embedding-nomic-embed-text-v1.5")
# embedding = TextEmbedEmbeddings(
#     api_url="http://localhost:8000/v1", model="text-embedding-nomic-embed-text-v1.5",api_key=""
# )
db= Chroma.from_documents(texts,embedding=embedding)

retriever = db.as_retriever(search_type="mmr",search_kwargs={"k":8})

result = retriever.invoke("read file from disk")
print(result)

from openai import OpenAI

from langchain.memory import ConversationSummaryMemory
from langchain.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


llm = ChatOpenAI(model="chatglm",  api_key="your_api_key",
                 base_url="http://localhost:8000/v1", temperature=0.01)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# from langchain.chains import (
#     StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
# )

# question_generator_chain = LLMChain(llm=llm, prompt=prompt)
# chain = ConversationalRetrievalChain(
#     combine_docs_chain=combine_docs_chain,
#     retriever=retriever,
#     question_generator=question_generator_chain,
# )

# qa = ConversationalRetrievalChain.from_llm()

