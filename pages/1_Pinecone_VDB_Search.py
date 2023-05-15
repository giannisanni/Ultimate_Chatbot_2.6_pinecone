import streamlit as st
from langchain.tools import human
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import shutil
from langchain.llms import OpenAIChat
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain import LLMChain, LLMMathChain
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain import GoogleSearchAPIWrapper
import os
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import load_tools
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools.ifttt import IFTTTWebhook
from langchain.utilities import PythonREPL
from langchain.utilities import BashProcess
api = "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" #st.sidebar.text_input("API-KEY", type="password")

# from api_key_gpt import API_KEYs
PINECONE_API_KEY = 'e7361a69-30fa-4a6d-b2e1-305f0b729d0e'
PINECONE_API_ENV = 'us-east4-gcp'

os.environ["GOOGLE_CSE_ID"] = "751a6a2ce2d114029"
os.environ['GOOGLE_API_KEY'] = "AIzaSyBPEpyGGYbApzNqvQ40t32zbFMXshcBcBU"
os.environ["WOLFRAM_ALPHA_APPID"] = "H342QE-H9X8JTY58A"

st.title('Pinecone Chatbot')


if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

# embeddings
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

# ifttt
key = os.environ["IFTTTKEY"] = "bFmxQ7aFnYyIveif2pbM3R"
url = f"https://maker.ifttt.com/trigger/spotify/json/with/kbFmxQ7aFnYyIveif2pbM3Rey/bFmxQ7aFnYyIveif2pbM3R/{key}"
spotify_skip = IFTTTWebhook(name="Spotify", description=" skip song spotify", url=url)

#retrieval
qa = RetrievalQA.from_chain_type(
    llm=OpenAIChat(openai_api_key=api),
    chain_type="stuff",
    retriever=index.as_retriever(),
)

llm = OpenAIChat(temperature=0)
# search engines
llm_math = LLMMathChain(llm=llm, verbose=True)
google_search = GoogleSearchAPIWrapper(k=1)
wolfram = WolframAlphaAPIWrapper()
wikipedia = WikipediaAPIWrapper()
python_repl = PythonREPL()
bash = BashProcess()
# my tools
tools = [
    Tool(
        name="Bash",
        func=bash.run,
        description= "for interacting with the operating system. It is commonly used for automating tasks, such as running scripts or batch processing files. Bash can also be used for system administration tasks, such as managing files and directories, setting environment variables, and controlling processes.", return_direct=True
    ),
    Tool(
        name="Answer Machine",
        func=qa.run,
        description="Useful for when you need to answer questions about uploaded documents, pdf, books and personal information about gianni. also use if input is in dutch. Input may be a partial or fully formed question."
                    " always use this tool when person is speaking in the first person", #return_direct=True
    ),
    Tool(
        name="Google Search",
        func=google_search.run,
        description="useful for when you need to answer questions about current events. you need to be specific. this can also be used for searching how to respond to greetings."
    ),
    Tool(
        name="Math",
        func=llm_math.run,
        description="Useful for when you need to answer questions about math. use for more simple calculations"
    ),
    Tool(
        name="Wolfram",
        func=wolfram.run,
        description="Useful for when you need to answer questions about science, Money, Finance, units, measurements, algebra, phydics, chemistry, etc. also use for complex calculations"
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to answer questions you need to look up in an encyclopedia ."
    ),
    Tool(
        name="Spotify Skip",
        func=spotify_skip.run,
        description="Useful for when you need to skip a current music track on spotify."
    ),
    Tool(
        name="Python REPL",
        func=python_repl.run,
        description="Useful for when you need to use python code."
    )
]
prefix = """Have a conversation with a human named Gianni (you are named Penny), the person you are talking too is gianni sanrochman, answering the following questions as best you can based on the context. 
            You have access to tools. If you use wikipedia tool, parse the outcome first.
            you should always give a final answer like this example: 
            Thought: Gianni is greeting me, I should respond politely.
Action: tool
Action Input: "........."
Observation:...........” 
Thought:I................."
Final Answer:..........."
If someone asks you to perform a task, your job is to come up with a series of bash commands that will perform the task. There is no need to put "#!/bin/bash" in your answer. Make sure to reason step by step, using this format:
Question: "copy the files in the directory named 'target' into a new directory at the same level as target called 'myNewDirectory'"
I need to take the following actions:
- List all files in the directory
- Create a new directory
- Copy the files from the first directory into the second directory
```bash
ls
mkdir myNewDirectory
cp -r target/* myNewDirectory
```

Do not use 'echo' when writing the script.

That is the format. Begin!"""

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
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=6)

# chain
# zero shot agent
# agent executor
llm_chain = LLMChain(
    llm=OpenAIChat(
        temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo",streaming=True
    ),
    prompt=prompt,
)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, max_iterations=3)
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


