#%% Import Libraries

import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

#%% Loading API Key

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
if 'STREAMLIT_PUBLIC_PATH' in os.environ:
    os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
    openai.api_key = os.getenv("OPENAI_API_KEY")

#%% Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [('system', "You are a walking Wikipedia, please answer the users' questions"),
    ('user', 'Question: {question}')])

#%% Generate Answers

def generate_reponse(question, model, temperature, max_tokens):
    
    llm = ChatOpenAI(model = model)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question': question})

    return answer

#%% Create Streamlit App

st.title('ChatBot with OpenAI')
st.sidebar.title('Settings')
llm_list = ['gpt-4o-mini', 'gpt-4o', 'o1-mini', 'gpt-3.5-turbo-0125']
llm = st.sidebar.selectbox("Select Model", llm_list, index=llm_list.index('gpt-4o-mini'))
st.sidebar.write(f'Selected MOdel: {llm}')
t = st.sidebar.slider('Temperature', min_value = 0.0, max_value = 1.0, value = .7)
max_tokens = st.sidebar.slider('Max Tokens', min_value = 150, max_value = 500, value = 250)

user_input = st.text_area("What's your question:")

if user_input:
    answer = generate_reponse(user_input, llm, t, max_tokens)
    st.write('### The answer is:')
    st.write(answer)
