import openai
import streamlit as st
import re
import time
import os
from io import BytesIO
from typing import Any, Dict, List
from langchain import LLMChain
from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from langchain import GoogleSearchAPIWrapper
from PIL import Image

# front_page
st.set_page_config(
    page_title="The Dreamer",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="auto",
)


# sidebar image
#def add_logo(logo_path, width, height):
    #"""Read and return a resized logo"""
    #logo = Image.open(logo_path)
    #modified_logo = logo.resize((width, height))
    #return modified_logo


#st.sidebar.image(
    #add_logo(logo_path='C:/Users/gsanr/Downloads/streamlit_sidebar_banner-image.png', width=300, height=80))

# password
api = "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" #st.sidebar.text_input("API-KEY", type="password")
# from api_key_gpt import API_KEYs
os.environ["LANGCHAIN_HANDLER"] = "langchain"
os.environ['SERPAPI_API_KEY'] = "0d1c57a0026ffa69769de5aab1a43def7f7b902b361b8d42989921d0ce540ae2"
os.environ['WOLFRAM_ALPHA_APPID'] = "H342QE-H9X8JTY58A"
os.environ["OPENAI_API_KEY"] = "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"#api
os.environ["APIFY_API_TOKEN"] = "apify_api_ZUT5kiG3E0jMsceVQOBJZ5Xnxrl8yn1zwDMb"
os.environ["GOOGLE_CSE_ID"] = "2028b6b5a7ab74bbe"
os.environ['GOOGLE_API_KEY'] = "AIzaSyCXl_MN9qS6rZgcqkX572nRGNdPSb7PujE"
# Engine_Swap
format_type = st.sidebar.selectbox('Choose your Dreamer',
                                   ["TURBO_GPT3.5", "Davinci", "DALL-E 2", "Mental Health", "Madam Curie", "InterBot",
                                    "FileMaster"])


# DALLE_2
@st.cache_data(persist=True, show_spinner=False)
def generate_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url


def create_image_variations(file, size="1024x1024"):
    response = openai.Image.create_variation(
        image=file,
        n=2,
        size=size
    )
    return response['data'][0]['url']


st.title("THE DREAMER")

# Chatbot_Davinci
if format_type == "Davinci":

    # initialize session states
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []


    @st.cache_data(experimental_allow_widgets=True, persist=True, show_spinner=False)
    def openai_completion():
        """
        Get the user input text.
        Returns:
            (str): user input text entered by the user
        """
        input_text = st.text_area("You: ", st.session_state["input"], key="input",
                                  placeholder="I ponder🤔",
                                  label_visibility="hidden")
        chat_button = st.button("🔥Let me cook🔥")
        if chat_button and input_text.strip() != "":
            with st.spinner("📝.📜.🧠.💭.💡"):
                return input_text


    if api:
        llm = OpenAI(
            model="text-davinci-003",
            openai_api_key=api,
            max_tokens=150,
            temperature=0.5,
        )
        #  conversational memory
        if "entity_memory" not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)

        # Conversation Chain
        Conversation = ConversationChain(
            llm=llm,
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
        )
    else:
        st.warning("Probeer opnieuw")

    # user input
    user_input = openai_completion()

    # generate output using conversation chain
    if user_input:
        output = Conversation.run(input=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    with st.expander("Chat History", expanded=True):
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            st.info(st.session_state["generated"][i], icon="🧙")
            st.write(st.session_state["past"][i])

# DAVINCI
if format_type == "Mental Health":
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []


    @st.cache_data(experimental_allow_widgets=True, persist=True, show_spinner=False)
    def openai_completion():
        """
        Get the user input text.
        Returns:
            (str): user input text entered by the user
        """
        input_text = st.text_area("You: ", st.session_state["input"], key="input",
                                  placeholder="Tell me how you're feeling",
                                  label_visibility="hidden")
        chat_button = st.button("let me think🤔")
        return input_text


    if api:
        llm = OpenAI(
            model="",
            openai_api_key=api,
            max_tokens=150,
            temperature=0.5,
        )
        if "entity_memory" not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
        Conversation = ConversationChain(
            llm=llm,
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
        )
    else:
        st.warning("Probeer opnieuw")
    user_input = openai_completion()
    if user_input:
        output = Conversation.run(input=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    with st.expander("Chat History", expanded=True):
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            st.info(st.session_state["generated"][i], icon="🧠")
            st.write(st.session_state["past"][i])

# MADAM CURIE
if format_type == "Madam Curie":
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []


    @st.cache_data(experimental_allow_widgets=True, persist=True, show_spinner=False)
    def openai_completion():
        """
        Get the user input text.
        Returns:
            (str): user input text entered by the user
        """
        input_text = st.text_area("You: ", st.session_state["input"], key="input",
                                  placeholder="How can i help?",
                                  label_visibility="hidden")
        chat_button = st.button("let me think🤔")
        return input_text


    if api:
        llm = OpenAI(
            model="",
            openai_api_key=api,
            max_tokens=150,
            temperature=0.5,
        )
        if "entity_memory" not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
        Conversation = ConversationChain(
            llm=llm,
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
        )
    else:
        st.warning("Probeer opnieuw")
    user_input = openai_completion()
    if user_input:
        output = Conversation.run(input=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    with st.expander("Chat History", expanded=True):
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            st.info(st.session_state["generated"][i], icon="🧠")
            st.write(st.session_state["past"][i])

# TURBO
if format_type == "TURBO_GPT3.5":
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []


    @st.cache_data(experimental_allow_widgets=True, persist=True, show_spinner=False)
    def openai_completion():
        """
        Get the user input text.
        Returns:
            (str): user input text entered by the user
        """
        input_text = st.text_area("You: ", st.session_state["input"], key="input",
                                  placeholder="How can i help?",
                                  label_visibility="hidden")
        chat_button = st.button("🚀GO!🚀")
        return input_text


    if api:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=api,
            max_tokens=150,
            temperature=0.5,

        )
        if "entity_memory" not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
        Conversation = ConversationChain(
            llm=llm,
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory,
        )
    else:
        st.warning("enter valid API in sidebar")
    user_input = openai_completion()
    if user_input:
        output = Conversation.run(input=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    with st.expander("Chat History", expanded=True):
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            st.info(st.session_state["generated"][i], icon="🤖")
            st.write(st.session_state["past"][i])

# interface dalle
if format_type == "DALL-E 2":
    input_text = st.text_area("Let me paint you a picture🎆", height=50)
    image_button = st.button("🌌💤Dream💤🌌")

    if image_button and input_text.strip() != "":
        with st.spinner("I am Dreaming💤💤"):
            image_url = generate_image(input_text)
            st.image(image_url, caption="Those who do not want to imitate anything, produce nothing.")
    else:
        st.warning("A true artist is not one who is inspired,"
                   " but one who inspires others")

    # Create a file uploader in the Streamlit app
    uploaded_file = st.file_uploader("Or Create a variation of your own image", type="png",
                                     accept_multiple_files=False,
                                     )

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Call the create_image_variations function with the uploaded file
        image_url = create_image_variations(uploaded_file)
        st.image(image_url)


# INTERBOT
def fib(n):
    if n <= 1:
        return n
    else:
        return (fib(n - 1) + fib(n - 2))


MODEL = "gpt-3.5-turbo"
llm = OpenAI(temperature=0, model_name=MODEL)

# serpapi
llm1 = ChatOpenAI(temperature=0)
llm2 = OpenAI(temperature=0)
#search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm1, verbose=True)
google_search = GoogleSearchAPIWrapper(k=1)
tools = [
    Tool(
        name="Google Search",
        func=google_search.run,
        description="useful for when you need to answer questions about current events. also ask follow up questions."
    ),
]

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

if format_type == "InterBot":
    # st.set_page_config(page_title="INTERBOT", page_icon=":robot:")
    st.header(" InterBOT")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
        st.session_state["stored_session"] = []


    def get_text():
        """
        Get the user input text.
        Returns:
            (str): The text entered by the user
        """
        input_text = st.text_input("You: ", st.session_state["input"], key="input")
        placeholder = "Your Ai assistant st your service",
        label_visibility = "hidden"
        return input_text


    # get user input
    user_input = get_text()
    if user_input:
        output = agent_chain.run(input=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            st.write(st.session_state["generated"][i], key=str(i))
            st.write(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


# personal files chatbot
if format_type == "FileMaster":
    #st.set_page_config(page_title="Chatbot")


    @st.cache_data
    def parse_pdf(file: BytesIO) -> List[str]:
        pdf = PdfReader(file)
        output = []
        for page in pdf.pages:
            text = page.extract_text()
            # Merge hyphenated words
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            # Fix newlines in the middle of sentences
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            # Remove multiple newlines
            text = re.sub(r"\n\s*\n", "\n\n", text)
            output.append(text)
        return output


    @st.cache_data
    def text_to_docs(text: str) -> List[Document]:
        """Converts a string or list of strings to a list of Documents
        with metadata."""
        if isinstance(text, str):
            # Take a single string as one page
            text = [text]
        page_docs = [Document(page_content=page) for page in text]

        # Add page numbers as metadata
        for i, doc in enumerate(page_docs):
            doc.metadata["page"] = i + 1

        # Split pages into chunks
        doc_chunks = []

        for doc in page_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=0,
            )
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
                )
                # Add sources a metadata
                doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
                doc_chunks.append(doc)
        return doc_chunks


    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        doc = parse_pdf(uploaded_file)
        pages = text_to_docs(doc)
        # pages
        if pages:
            with st.expander("page Content", expanded=False):
                page_sel = st.number_input(
                    label="select a page", min_value=1, max_value=len(pages), step=1
                )
                pages[page_sel - 1]
            #api = st.sidebar.text_input(
                #"Enter OpenAI API Key",
                #type="password",
                #placeholder="sk-00000",
                #help="https://platform.openai.com/account/api-keys",
            #)
            if api:
                embeddings = OpenAIEmbeddings(openai_api_key=api)
                # indexing
                # save in a VECTOR DB
                with st.spinner("Indexing..."):
                    index = FAISS.from_documents(pages, embeddings)

                qa = RetrievalQA.from_chain_type(
                    llm=OpenAI(openai_api_key=api),
                    chain_type="stuff",
                    retriever=index.as_retriever(),
                )

                # my tools
                tools = [
                    Tool(
                        name="Answer Machine",
                        func=qa.run,
                        description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                    )
                ]

                prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available.
                            You have access to a single tool:"""
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
                    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

                # chain
                # zero shot agent
                # agent executor
                llm_chain = LLMChain(
                    llm=OpenAI(
                        temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
                    ),
                    prompt=prompt,
                )

                agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
                agent_chain = AgentExecutor.from_agent_and_tools(
                    agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
                )

                query = st.text_input("Ask away...")
                if query:
                    res = agent_chain.run(query)
                    st.write(res)

            with st.expander("In recent memory", expanded=False):
                st.session_state.memory

            # with st.expander("In recent memory", expanded=False):
            # st.session_state.memory


hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

st.sidebar.markdown("GIANNISAN")
