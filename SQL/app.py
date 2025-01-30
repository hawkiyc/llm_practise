#%% Import Libraries

import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

#%% Load API Key

if 'STREAMLIT_PUBLIC_PATH' in os.environ:
    # Deploy on Streamlit Cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    # Deploy on local
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')

#%% Build the App

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

db_opt = ['SQLite', 'MySQL']

db = st.sidebar.radio("Chosse Database you want to connect", db_opt,)

if db == 'MySQL':
    db_host = st.sidebar.text_input("Provide MySQL Host:")
    db_usr = st.sidebar.text_input("MySQL Username:")
    db_pw = st.sidebar.text_input("MySQL Password:", type='password')
    
    