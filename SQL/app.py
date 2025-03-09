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
import mysql.connector
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

#%% Setting Database Parameters

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

db_opt = ['SQLite', 'MySQL']

selected_db = st.sidebar.radio("Chosse Database you want to connect", db_opt,)

if selected_db == 'MySQL':
    db_host = st.sidebar.text_input("Provide MySQL Host:")
    db_usr = st.sidebar.text_input("MySQL Username:")
    db_pw = st.sidebar.text_input("MySQL Password:", type='password')
    
    if db_host and db_usr and db_pw:
        try:
            # connect to MySQL and fetch all database
            conn = mysql.connector.connect(
                host=db_host, user=db_usr, password=db_pw)
            cursor = conn.cursor()
            cursor.execute("SHOW DATABASES")
            databases = [db[0] for db in cursor.fetchall()]
            cursor.close()
            conn.close()
            if databases:
                mysql_db = st.sidebar.radio("Select a MySQL Database", databases)
            else:
                st.sidebar.warning("No databases found in MySQL server.")
        except Exception as e:
            st.error(f"MySQL connection failed: {e}")
elif selected_db == 'SQLite':
    db_files = list(Path(".").rglob("*.db"))  # search all splite db files
    db_file_map = {db.name: db.resolve() for db in db_files} 

    if db_file_map:
        selected_db_file = st.sidebar.radio("Select a SQLite Database", list(db_file_map.keys()))
        selected_db_file = db_file_map[selected_db_file]
    else:
        st.sidebar.warning("No SQLite database files found.")

if not selected_db:
    st.info("Please select the database you want to connect.")

if selected_db == "MySQL" and not mysql_db:
    st.warning("Please select a MySQL database to proceed.")

if selected_db == "SQLite" and not selected_db_file:
    st.warning("Please select a SQLite database to proceed.")

@st.cache_resource(ttl="2h")
def configure_db(selected_db, db_host=None, db_usr=None, db_pw=None, mysql_db=None, sqlite_db_path=None):
    if selected_db == 'SQLite' and sqlite_db_path:
        dbfilepath = Path(sqlite_db_path).absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif selected_db == 'MySQL':
        if not (db_host and db_usr and db_pw and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{db_usr}:{db_pw}@{db_host}/{mysql_db}"))   

if selected_db == 'MySQL' and db_host and db_usr and db_pw and mysql_db:
    db = configure_db(selected_db, db_host, db_usr, db_pw, mysql_db)
elif selected_db == "SQLite" and selected_db_file:
    db = configure_db(selected_db, sqlite_db_path=selected_db_file)
else:
    db = None

#%% Build Toolkit and Agent

system_prompt = """
You are an AI assistant specialized in executing SQL queries. 
Follow these strict guidelines:
1. Always return an action in a structured format, never provide a natural language explanation.
2. If you need to perform a database action, return only the structured action and avoid additional text.
3. Never mix an answer with an action. If an action is needed, do not include a final answer.
"""

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", streaming=True)
if db:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        if msg["role"] != "system": 
            st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask anything from the database")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)   