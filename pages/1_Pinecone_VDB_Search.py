import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAIChat
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain import LLMChain, LLMMathChain
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain import GoogleSearchAPIWrapper
import os
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
api = "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" #st.sidebar.text_input("API-KEY", type="password")
# from api_key_gpt import API_KEYs
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]

os.environ["GOOGLE_CSE_ID"] = "2028b6b5a7ab74bbe"
os.environ['GOOGLE_API_KEY'] = "AIzaSyCXl_MN9qS6rZgcqkX572nRGNdPSb7PujE"


st.title('Pinecone Chatbot')

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

embeddings = OpenAIEmbeddings(openai_api_key=api)
# indexing

# save in a VECTOR DB
with st.spinner("Indexing..."):
    index_name = "pineconevoicebot"
    index = Pinecone.from_texts('', embeddings, index_name=index_name)
    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )

qa = RetrievalQA.from_chain_type(
    llm=OpenAIChat(openai_api_key=api),
    chain_type="stuff",
    retriever=index.as_retriever(),
)

llm = OpenAIChat(temperature=0)
llm_math = LLMMathChain(llm=llm, verbose=True)
google_search = GoogleSearchAPIWrapper(k=1)

# my tools
tools = [
    Tool(
        name="Answer Machine",
        func=qa.run,
        description="Useful for when you need to answer questions about uploaded documents. Input may be a partial or fully formed question."
    ),
    Tool(
        name="Google Search",
        func=google_search.run,
        description="useful for when you need to answer questions about current events."
    ),
    Tool(
        name="Math",
        func=llm_math.run,
        description="Useful for when you need to answer questions about math."
    )
]

prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available.
            You have access to tools. you also like to use slang words"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)

# chain
# zero shot agent
# agent executor
llm_chain = LLMChain(
    llm=OpenAIChat(
        temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo",streaming=True,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    ),
    prompt=prompt,
)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
)

query = st.text_input("Ask away...")
#if query:
    #res = agent_chain.run(query)
    #st.success(res)

with st.expander("In recent memory", expanded=False):
    st.session_state.memory

if query:
    res = agent_chain.run(query)
    output = qa(res)
    # store the output
    # Store conversation history
    st.session_state.past.append(query)
    st.session_state.generated.append(res)


if st.session_state['generated']:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.success(st.session_state["generated"][i], icon="🤖")
        st.info(st.session_state["past"][i])



    #for i in range(len(st.session_state['generated'])-1, -1, -1):
       # message(st.session_state["generated"][i], key=str(i))
       # message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# with st.expander("In recent memory", expanded=False):
# st.session_state.memory
