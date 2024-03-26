from llamaapi import LlamaAPI

# Replace 'Your_API_Token' with your actual API token
llama = LlamaAPI("LL-I05pgy3RKmwMNw8Lq2PJqdqC1dTDJzsYkzlXgLCK1kA3XEqLFSHoUO4tnFH6SF08")

from langchain_experimental.llms import ChatLlamaAPI

model = ChatLlamaAPI(client=llama)
from langchain.chains import create_tagging_chain

schema = {
    "properties": {
        "sentiment": {
            "type": "string",
            "description": "the sentiment encountered in the passage",
        },
        "aggressiveness": {
            "type": "integer",
            "description": "a 0-10 score of how aggressive the passage is",
        },
        "language": {"type": "string", "description": "the language of the passage"},
    }
}

chain = create_tagging_chain(schema, model)
rep = chain.invoke("give me your money")
print(rep)