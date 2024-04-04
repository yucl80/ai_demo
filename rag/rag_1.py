#https://lightning.ai/lightning-ai/studios/document-search-and-retrieval-using-rag

from typing import List, Optional, Union

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from rich import print
from sentence_transformers import CrossEncoder
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
from langchain.retrievers import ContextualCompressionRetriever

def rerank_docs(reranker_model, query, retrieved_docs):
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)


def load_pdf(
    files: Union[str, List[str]] = "example_data/2401.08406.pdf"
) -> List[UnstructuredFileLoader]:
    if isinstance(files, str):
        loader = UnstructuredFileLoader(
            files,
            post_processors=[clean_extra_whitespace, group_broken_paragraphs],
        )
        return [loader]

    loaders = [
        UnstructuredFileLoader(
            file,
            post_processors=[clean_extra_whitespace, group_broken_paragraphs],
        )
        for file in files
    ]
    return loaders


def split_text(
    loaders: List[UnstructuredFileLoader],
    separators=["\n\n\n", "\n\n"],
    chunk_size=1000,
):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    docs = []
    for loader in loaders:
        docs.extend(
            loader.load_and_split(text_splitter=text_splitter),
        )
    return docs


def load_embedding_model(
    model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda"
) -> HuggingFaceBgeEmbeddings:
    model_kwargs = {"device": device}
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding_model


def load_reranker_model(
    reranker_model_name: str = "BAAI/bge-reranker-large", device: str = "cuda"
) -> CrossEncoder:
    reranker_model = CrossEncoder(
        model_name=reranker_model_name, max_length=512, device=device
    )
    return reranker_model


def generate_embeddings(docs, embedding_model):
    db = FAISS.from_documents(documents=docs, embedding=embedding_model)
    return db


def create_compression_retriever(
    base_retriever, reranker_model
) -> ContextualCompressionRetriever:
    embeddings_filter = EmbeddingsFilter(
        embeddings=reranker_model, similarity_threshold=0.5
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=base_retriever
    )
    return compression_retriever


def main(file: str = "example_data/2401.08406.pdf", query: Optional[str] = None):
    loader = load_pdf(files=file)
    documents = split_text(
        loaders=loader,
    )

    embedding_model = load_embedding_model()
    reranker_model = load_reranker_model()

    vectorstore = generate_embeddings(documents, embedding_model=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    retrieved_documents = retriever.get_relevant_documents(query)
    return rerank_docs(reranker_model, query, retrieved_documents)
