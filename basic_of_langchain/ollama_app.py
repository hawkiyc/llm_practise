#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:12:00 2024

@author: revlis_ai
"""

#%%
"Import Libraries"

import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

#%%
"Load Env Parameter"

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

#%%
"LLM APP"

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a experienced cardio-doctor. Please answer patient's question"),
     ("user","Question:{question}")])

st.title("Langchain Demo with LLAMA")
input_text = st.text_input("Please enter your question:")

llm = Ollama(model = 'llama3.2')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))

