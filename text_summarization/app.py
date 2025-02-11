# Import Libraries

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Load API Key

if 'STREAMLIT_PUBLIC_PATH' in os.environ:
    # Deploy on Streamlit Cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    # Deploy on local
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')

# Build Streamlit App

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768", )
chunk_prompt = """Please provide a summary of the following content:
    Content: {text}
    """
chunk_prompt = PromptTemplate(input_variables=["text"], template=chunk_prompt)
final_prompt = """
    Please provide the final summary with several important sections of the given content:
    Content:{text}"""
final_prompt = PromptTemplate(template=final_prompt, input_variables=["text"])

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')
target_url = st.text_input("Enter URL to Summarize", placeholder="https://en.wikipedia.org/wiki/Siege_of_Baghdad")

if st.button("Summarize"):
    if not target_url or not validators.url(target_url):
        st.warning("Please enter a valid URL")
    else:
        try:
            with st.spinner("Summarizing..."):
                if "youtube.com" in target_url:
                    loader = YoutubeLoader(target_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[target_url], ssl_verify=False, headers=headers,)
                docs = loader.load()
                splitted_docs = RecursiveCharacterTextSplitter(chunk_size = 2048, chunk_overlap = 128).split_documents(docs)
                chain = load_summarize_chain(
                    llm, chain_type='map_reduce', map_prompt=chunk_prompt, combine_prompt=final_prompt,verbose=False)
                summary = chain.run(splitted_docs)
                st.success(summary)
        except Exception as e:
            st.exception(f'Exception: {e}')
