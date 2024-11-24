#%% Import Libraries

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache()

#%% Load API Key

if 'STREAMLIT_PUBLIC_PATH' in os.environ:
    # Deploy on Streamlit Cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["HF_TOKEN"] = st.secrets['HUGGINGFACE_TOKEN']
else:
    # Deploy on local
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    os.environ["HF_TOKEN"] = os.getenv('HUGGINGFACE_TOKEN')

#%% Build ChatBot with RAG

embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2',)
llm = ChatGroq(api_key = groq_api_key, model = "llama-3.2-90b-text-preview")

st.title('ChatBot Q&A with RAG')
st.write('Upload Paper in PDF Format and Ask Your Question')
session_id = st.text_input('Session ID:', value = 'your id')

if 'store' not in st.session_state:
    st.session_state.store = {}

upload_pdf_files = st.file_uploader('Upload PDF Files', type=['pdf'], accept_multiple_files=True)

if upload_pdf_files:

    st.success('Files are uploaded!!!')
    documents = []
    for f in upload_pdf_files:
        temp = './temp.pdf'
        with open(temp, 'wb') as file:
            file.write(f.getvalue())
        loader = PyPDFLoader(temp)
        doc = loader.load()
        documents.extend(doc)
        os.remove(temp)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 5140, chunk_overlap = 512)
    chunked_documents = splitter.split_documents(documents)
    chroma_db = Chroma.from_documents(documents = chunked_documents, embedding = embeddings)
    retriever = chroma_db.as_retriever()

    contextual_sys_prompt = '''
        Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is.
        '''
    contextual_prompt = ChatPromptTemplate.from_messages(
        [('system', contextual_sys_prompt), 
         MessagesPlaceholder('chat_history'),
         ('human', '{input}')])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextual_prompt)

    sys_prompt = '''
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
        \n\n
        {context}'''
    qa_prompt = ChatPromptTemplate.from_messages(
        [('system', sys_prompt),
         MessagesPlaceholder('chat_history'),
         ('human', '{input}')])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",)
    
    user_input = st.text_area("Ask me your question:")
    if user_input:
        session_history = get_session_history(session_id)
        ans = conversational_rag_chain.invoke(
            {'input': user_input},
            config = {'configurable': {'session_id': session_id}})
        st.write("Assistant:", ans['answer'])
        st.write("Chat History:",)
        for message in st.session_state.store[session_id].messages:
            st.write(f"Role: {message.type}, Content: {message.content}")
else:
    st.write('Please Upload Your Files!!!')
    