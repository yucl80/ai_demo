from langchain_community.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace

# loader = UnstructuredFileLoader("/home/test/src/state_of_the_union.txt")

loader = UnstructuredFileLoader(
    "/home/test/src/layout-parser-paper.pdf", mode="elements",
    post_processors=[clean_extra_whitespace],
)

docs = loader.load()

print(docs)
