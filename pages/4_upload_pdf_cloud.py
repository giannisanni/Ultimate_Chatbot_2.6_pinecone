import streamlit as st
from pathlib import Path
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Chroma, Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

st.title('Upload PDF file..')

pdf_file_uploaded = st.file_uploader(label="Your PDF file")

OPENAI_API_KEY = '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'
PINECONE_API_KEY = 'e7361a69-30fa-4a6d-b2e1-305f0b729d0e'
PINECONE_API_ENV = 'us-east4-gcp'

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "pineconevoicebot"

if pdf_file_uploaded is not None:
    loader = UnstructuredPDFLoader(pdf_file_uploaded)

    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)