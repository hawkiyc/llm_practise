#%%
"Import Libraries"

import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()


#%%
"Load API Key"

groq_api_key = os.getenv('GROQ_API_KEY')

#%%
"Build App"

llm = ChatGroq(api_key = groq_api_key, model = "llama-3.2-11b-text-preview")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """)

def create_embedding_vector():

    if "vectors" not in st.session_state:
        st.session_state.embedding = OllamaEmbeddings(model = "llama3.2")
        try:
            st.session_state.vectors = FAISS.load_local("./FAISS_DB", st.session_state.embedding, allow_dangerous_deserialization = True)
            print('Loading FAISS DB!!!')
        except:
            print('Create FAISS DB!!!')
            st.session_state.loader = PyPDFDirectoryLoader(path = './research_papers')
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 256)
            st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
            FAISS_DB = FAISS.from_documents(st.session_state.final_docs, st.session_state.embedding)
            st.session_state.vectors = FAISS_DB
            FAISS_DB.save_local("FAISS_DB")
            print("FAISS DB Saved!!!")

st.title('RAG Q&A with Groq and Llama3.2')

user_prompt = st.text_input("Enter your questions about the research articles")

if st.button("Docs Embedding"):
    create_embedding_vector()
    st.write("Vector DB is Ready!!!")

if user_prompt:
    docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, docs_chain)
    reponse = retrieval_chain.invoke({'input' : user_prompt})
    st.write(reponse['answer'])

    # Streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(reponse['context']):
            st.write(doc.page_content)
            st.write("-----------------------------------")
