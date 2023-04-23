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

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "pineconevoicebot"

if pdf_file_uploaded is not None:
        
    def save_file_to_folder(uploadedFile):
            # Save uploaded file to 'content' folder.
            save_folder = 'C:/Users/gsanr/OneDrive/Desktop/filefolder'
            save_path = Path(save_folder, uploadedFile.name)
            with open(save_path, mode='wb') as w:
                w.write(uploadedFile.getvalue())

            if save_path.exists():
                st.success(f'File {uploadedFile.name} is successfully saved!')
                
    save_file_to_folder(pdf_file_uploaded)
    
    loader = UnstructuredPDFLoader(f"C:/Users/gsanr/OneDrive/Desktop/filefolder/{pdf_file_uploaded.name}")

    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

   