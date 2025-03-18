#%% Import libraries

import requests
import json
import gradio as gr

#%% Define the function

url="http://localhost:11434/api/generate"
headers = {'Content-Type': 'application/json'}
history = []

def generate_response(text):
    history.append(text)
    final_text = "\n".join(history)
    data = {'model': 'CodeAssistantBeta', 'prompt': final_text, 'stream': False,}
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        data = response.json()
        return data['response']
    else:
        print("Error:", response.text)

interface = gr.Interface(
    fn=generate_response, 
    inputs=gr.Textbox(lines=6,placeholder='Ask me your question'), 
    outputs="text", title="Code Assistant", 
    description="This is a code assistant that helps you write code . Just ask your question and I will help you complete your code.")
interface.launch()

