#%% Import Libraries

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

#%% Load API Key

if 'STREAMLIT_PUBLIC_PATH' in os.environ:
    # Deploy on Streamlit Cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    # Deploy on local
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')

#%% Load LLM

llm = ChatGroq(model="llama-3.2-90b-text-preview", api_key=groq_api_key)
sys_temp = "You are an experienced translator; please translate English input into {language}"
prompt = ChatPromptTemplate.from_messages(
    [('system', sys_temp), ('user', '{text}')])
parser = StrOutputParser()
chain = prompt | llm | parser

#%% Streamlit Interface

st.title("Langchain Translator")
st.write("Translate text to your chosen language.")

# Language and text input
language = st.text_input("Enter Target Language", value="Traditional Chinese")
text = st.text_area("Enter text to translate")

if st.button("Translate"):
    # Update the prompt with the chosen language and input text using chain.invoke()
    translated_text = chain.invoke({"language": language, "text": text})
    
    # Display the translated text
    st.write("### Translated Text")
    st.write(translated_text)
