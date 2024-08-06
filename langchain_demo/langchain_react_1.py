import os
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults

# set up API key
os.environ["TAVILY_API_KEY"] = "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"

# set up the agent
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7,base_url="http://localhost:8000/v1/", api_key="nokey")
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

# initialize the agent
agent_chain = initialize_agent(
    [tavily_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# run the agent
result = agent_chain.run(
    "What happened in the latest burning man floods?",
)
print(result)


# Step 0. Importing relevant Langchain libraries
# from langchain.adapters.openai import convert_openai_messages
# from langchain_community.chat_models import ChatOpenAI

# Step 1. Instantiating your TavilyClient
# from tavily import TavilyClient
# client = TavilyClient(api_key="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")

# # Step 2. Executing the search query and getting the results
# content = client.search("What happened in the latest burning man floods?", search_depth="advanced")["results"]

# # Step 3. Setting up the OpenAI prompts
# prompt = [{
#     "role": "system",
#     "content":  f'You are an AI critical thinker research assistant. '\
#                 f'Your sole purpose is to write well written, critically acclaimed,'\
#                 f'objective and structured reports on given text.'
# }, {
#     "role": "user",
#     "content": f'Information: """{content}"""\n\n' \
#                f'Using the above information, answer the following'\
#                f'query: "{query}" in a detailed report --'\
#                f'Please use MLA format and markdown syntax.'
# }]

# # Step 4. Running OpenAI through Langchain
# lc_messages = convert_openai_messages(prompt)
# report = ChatOpenAI(model='gpt-4',openai_api_key="sk-YOUR_OPENAI_KEY").invoke(lc_messages).content

# # Step 5. That's it! Your research report is now done!
# print(report)