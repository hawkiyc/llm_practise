#%% Import Libraries

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

#%% Load API Key

if 'STREAMLIT_PUBLIC_PATH' in os.environ:
    # Deploy on Streamlit Cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    # Deploy on local
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')

#%% Build the App

st.set_page_config(page_title="LangChain: Solve Math Questions", page_icon="🧮")
st.title("🧮 LangChain: Solve Math Questions")

llm = ChatGroq(api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b", )

wiki_wrapper = WikipediaAPIWrapper(top_k_results = 3,doc_content_chars_max = 512)
wiki_tool = Tool(
    name="Wikipedia", 
    func=wiki_wrapper.run, 
    description='A tool for searching the Internet to find the various information on the topics mentioned')

math_chain = LLMMathChain.from_llm(llm = llm,)
math_tool = Tool(
    name="Math Solver", 
    func=math_chain.run, 
    description='A tool for solving math problems')

prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt)

chain = LLMChain(llm = llm, prompt = prompt_template,)

reasioning_tool = Tool(
    name="Reasoning", 
    func=chain.run, 
    description='A tool for reasoning the math problems')

agent = initialize_agent(
    tools=[wiki_tool, math_tool, reasioning_tool],
    llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False, handle_parsing_errors=True, )

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant', 
         'content': "I am an AI assistant that can help you solve math questions. Please ask me a math question."}]

for msg in st.session_state.messages:
    if msg["role"] != "system": 
        st.chat_message(msg["role"]).write(msg["content"])

# def generate_response(question):
#     response = agent.invoke({'input': question})
#     return response

question = st.text_area("Ask me a math question:", "x+y=10, x-y=2, what is x and y?")

if st.button("Submit"):
    if question:
        with st.spinner("Thinking..."):
            
            st.session_state.messages.append({'role': 'user', 'content': question})
            st.chat_message("user").write(question)
            callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = agent.run(st.session_state.messages, callbacks=[callback])
            # response = generate_response(question)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            # st.success(response)
            st.chat_message("assistant").write(response)
    else:
        st.warning("Please ask me a question.")