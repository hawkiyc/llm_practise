#%% Import Libraries

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv

load_dotenv()

#%% Load API Key

groq_api_key = os.getenv('GROQ_API_KEY')

#%% Load LLM

llm = ChatGroq(model="llama-3.2-90b-text-preview", api_key = groq_api_key,)
sys_temp = "You are a expericnced translator, please translate English input into {language}"
prompt = ChatPromptTemplate.from_messages(
    [('system', sys_temp), ('user', '{text}')])
parser = StrOutputParser()
chain = prompt|llm|parser

#%% Create App

app = FastAPI(title = 'Langchain Server', version = "beta 0", description = 'A Simple API for Langchain Runable Interface')
add_routes(app = app, runnable = chain, path = '/chain')

#%% Execute App

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app = app, host = "127.0.0.1", port = 8000)
