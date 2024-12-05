#%% Import Libraries

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

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

#%% API Wrapper and Tools

arxiv_wrapper = ArxivAPIWrapper(top_k_results = 5,doc_content_chars_max = 512)
arxiv_tool = ArxivQueryRun(api_wrapper = arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results = 5,doc_content_chars_max = 512)
wiki_tool = WikipediaQueryRun(api_wrapper = wiki_wrapper)

search_wrapper = DuckDuckGoSearchAPIWrapper(safesearch = 'off', max_results = 5,)
search_tool = DuckDuckGoSearchResults(api_wrapper = search_wrapper,)

tools = [arxiv_tool, wiki_tool, search_tool]

#%% Build Streamlit App

st.write('ChatBot with Search Engine!!!!')

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant', 
         'content': 'I am a chatbot with search engine. I am here to answer your question. What do you want to know?'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role'].write(msg['content']))

