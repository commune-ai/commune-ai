#from example.examples import HelloWorld_2_0
from helper.compiler import register, gradio_compile

# from gradioWrapper import register, gradio_compile
import gradio as gr

@gradio_compile
class Greeting:

    @register(inputs=[gr.Textbox(label="name")], outputs=['text'])
    def Hello_World(self, name):
        return f"Hello {name}, and welcome to Gradio Flow 🤗" 


if __name__ == "__main__":
    Greeting().run()