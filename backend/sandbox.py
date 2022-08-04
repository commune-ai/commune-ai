import gradio as gr
from ray import serve
from transformers import pipeline
import requests

import ray
from ray import serve

ray.init(address='auto', namespace='sandbox')

serve.start()


@serve.deployment
def model(request):
    language_model = pipeline("text-generation", model="gpt2")
    query = request.query_params["query"]
    return language_model(query, max_length=100)

model.deploy()
example = "What's the meaning of life?"
response = requests.get(f"http://localhost:8000/model?query={example}")
print(response.text)

def gpt2(query):
    response = requests.get(f"http://localhost:8000/model?query={query}")
    return response.json()[0]["generated_text"]

iface = gr.Interface(
    fn=gpt2,
    inputs=[gr.inputs.Textbox(
        default=example, label="Input prompt"
    )],
    outputs=[gr.outputs.Textbox(label="Model output")]
)
iface.launch(server_name='0.0.0.0')
