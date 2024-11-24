#%% Import Libraries

import streamlit as st
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb.api
from chromadb.config import Settings

chromadb.api.client.SharedSystemClient.clear_system_cache()

#%% Get Correct Path

current_dir = Path(__file__).parent.absolute()
doc_dir = os.path.join(current_dir, "research_papers")
db_path = os.path.join(current_dir, "FAISS_DB")

#%% Load API Key

if 'STREAMLIT_PUBLIC_PATH' in os.environ:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["HF_TOKEN"] = st.secrets['HUGGINGFACE_TOKEN']
else:
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    os.environ["HF_TOKEN"] = os.getenv('HUGGINGFACE_TOKEN')

#%% Load LLM and Embedding

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
llm = ChatGroq(api_key=groq_api_key, model="llama-3.2-90b-text-preview")

#%% Initialize session state

if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
    st.session_state.current_session = None
    st.session_state.session_history = {}

#%% Build Streamlit APP

st.title('ChatBot Q&A with RAG')
st.sidebar.title("Conversations")

def new_chat():
    session_id = str(uuid4()).replace('-', '')
    st.session_state.conversations[session_id] = {
        "history": ChatMessageHistory(),
        "uploaded_files": []}
    st.session_state.current_session = session_id

if not st.session_state.conversations:
    new_chat()

if st.sidebar.button("New Chat"):
    new_chat()

session_ids = list(st.session_state.conversations.keys())
session_names = [f"Session {i+1}" for i in range(len(session_ids))]
selected_session = st.sidebar.radio(
    "Select Conversation",
    session_names,
    index=session_ids.index(st.session_state.current_session))

if selected_session:
    selected_index = session_names.index(selected_session)
    st.session_state.current_session = session_ids[selected_index]

current_session = st.session_state.current_session
conversation_data = st.session_state.conversations[current_session]

uploaded_files = st.file_uploader(
    'Upload PDF Files',
    type=['pdf'],
    accept_multiple_files=True,
    key=st.session_state.current_session)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.conversations[session_id]["history"]

if uploaded_files:
    documents = []
    for f in uploaded_files:
        if f.name not in conversation_data["uploaded_files"]:
            conversation_data["uploaded_files"].append(f.name)
            temp = './temp.pdf'
            with open(temp, 'wb') as file:
                file.write(f.getvalue())
            loader = PyPDFLoader(temp)
            doc = loader.load()
            documents.extend(doc)
            os.remove(temp)

    splitter = RecursiveCharacterTextSplitter(chunk_size=5140, chunk_overlap=512)

    try:
        st.write('Trying to load FAISS BD')
        faiss_db = FAISS.load_local(
            db_path + f"/{current_session}", 
            embeddings, 
            allow_dangerous_deserialization = True)
        if documents:
            st.write('Detect new file(s) upload, add docs into Chroma DB')
            chunked_documents = splitter.split_documents(documents)
            faiss_db.add_documents(chunked_documents)
            faiss_db.save_local(db_path + f"/{current_session}")    
        st.write('FAISS DB loaded!!!')
    except:
        st.write('No FAISS DB found, creating FAISS DB.......')
        chunked_documents = splitter.split_documents(documents)
        faiss_db = FAISS.from_documents(chunked_documents, embeddings)
        faiss_db.save_local(db_path + f"/{current_session}")
        st.write('FAISS DB Created!!!')

st.write(f"Uploaded Files: {conversation_data['uploaded_files']}")

user_input = st.text_area("Ask me your question:", key="input_text",)
if st.button("Submit"):
    if user_input:

        faiss_db = FAISS.load_local(
            db_path + f"/{current_session}", 
            embeddings, 
            allow_dangerous_deserialization = True)
        retriever = faiss_db.as_retriever()

        contextual_sys_prompt = '''
            Given a chat history and the latest user question which might reference context in the chat history, 
            formulate a standalone question which can be understood without the chat history. Do NOT answer the question, '''
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

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",)
        st.session_state.session_history[current_session] = get_session_history(current_session)
        ans = conversational_rag_chain.invoke(
            {'input': user_input},
            config={'configurable': {'session_id': current_session}})
        st.write("Assistant:", ans['answer'])

if current_session in st.session_state.session_history:
    st.write("Chat History:")
    for message in st.session_state.session_history[current_session].messages:
        st.write(f"Role: {message.type}, Content: {message.content}")
else:
    st.write("No chat history available.")
