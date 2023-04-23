import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAIChat
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

st.title('Pinecone Chatbot')



OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "pineconevoicebot"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

docsearch = Pinecone.from_texts('', embeddings, index_name=index_name)


@st.cache_data
def vector_index_query(question):
    llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0.1, openai_api_key=OPENAI_API_KEY)


    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(question, include_metadata=True)
    return chain.run(input_documents=docs, question=question)

    # Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


# We will get the user's input by calling the get_text function

def get_text():
    input_text = st.text_input("You: ", placeholder="Questions About Your Document?", key="input", label_visibility="hidden")
    return input_text
user_input = get_text()

if user_input:
    output = vector_index_query(user_input)
    # store the output
    # Store conversation history
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
