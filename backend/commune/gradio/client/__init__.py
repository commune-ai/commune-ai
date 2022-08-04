import os, sys
sys.path.append(os.environ["PWD"])

import requests
import gradio as gr
import socket
import random
from inspect import getfile

from commune.gradio.api.module import GradioAPI
from commune.process.base import BaseProcess
from commune.utils.misc import SimpleNamespace



class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    


class GradioClient:
    api = GradioAPI()

    @staticmethod
    def compile(self, live=False, flagging='never', theme='default' **kwargs):
        print("Just putting on the finishing touches... 🔧🧰")
        for func in GradioModule.find_registered_functions:
            registered_fn = getattr(self, func)
            registered_fn()

        demos = []
        names = []
        for fn_key, fn_params in self.registered_gradio_functons.items():                
            names.append(func)
            demos.append(gr.Interface(fn=getattr(self, fn_key),
                                        inputs=fn_params['inputs'],
                                        outputs=fn_params['outputs'],
                                        theme='default'))
            print(f"{func}....{bcolor.BOLD}{bcolor.OKGREEN} done {bcolor.ENDC}")

        print("\nHappy Visualizing... 🚀")
        return gr.TabbedInterface(demos, names)
    

    @staticmethod
    def run(self,host=None, port=None, live=False, replace=True, **kwargs):

        port = port if port else GradioModule.api.suggest_port()
        host = host if host else GradioModule.api.host
        url = SimpleNamespace({ 'api': f'http://{host}:{GradioModule.api.port}',
                              'gradio' : f'http://{host}:{port}'})
        

        allow_flagging = kwargs.get('flagging', 'never')

        gradio_metadata = { 
                           "host" : host, 
                           'url': url.gradio,
                           "live": live,
                           "port": port,
                           "module" : getfile(self.__class__), 
                           "name" : self.__class__.__name__,
                            **kwargs
                            }
        try:
            GradioModule.compile(live=live, allow_flagging=allow_flagging).launch(server_port=port) 
        except Exception:
            print(f"**{bcolor.BOLD}{bcolor.FAIL}CONNECTION ERROR{bcolor.ENDC}** 🐛The listening api is either not up or you choose the wrong port.🐛")
            return

        GradioModule.api.add_module(port='gradio/add', json=gradio_metadata)




    @staticmethod
    def find_registered_functions(self):
        '''
        find the registered functions
        '''
        return GradioModule.api.find_registered_functions()


    @staticmethod
    def register(inputs, outputs):
        def register_gradio(func):
            def wrap(self, *args, **kwargs):            
                try:
                    self.registered_gradio_functons
                except AttributeError:
                    print("✨Initializing Class Functions...✨\n")
                    self.registered_gradio_functons = dict()

                fn_name = func.__name__ 
                if fn_name in self.registered_gradio_functons: 
                    result = func(self, *args, **kwargs)
                    return result
                else:
                    self.registered_gradio_functons[fn_name] = dict(inputs=inputs, outputs=outputs)
                    return None
            wrap.__decorator__ = GradioModule.register
            return wrap
        return register_gradio






if __name__== '__main__':
    import streamlit as st

    class HelloWorld_2_0:

        def __init__(self):
            GradioModule.run(self)

        @gradioClient.register(inputs=["text", "text", gr.Radio(["morning", "evening", "night"])], outputs="text")
        def Hello(self, Lname : str, Fname : str, day : 'list[any]'=["morning", "evening", "night"]) -> str:
            return "Hello, {} {}".format(Fname, Lname)  

        @gradioClient.register(inputs=["text", "text"], outputs="text")
        def goodbye(self, Fname : str, Lname : str) -> str:
            return "Goodbye, {} {}".format(Fname, Lname)  
        
        @gradioClient.register(inputs=["text", gr.Checkbox() , gr.Slider(0, 60)], outputs=["text", "number"])
        def greet(self, name, is_morning, temperature):
            salutation = "Good morning" if is_morning else "Good evening"
            greeting = "%s %s. It is %s degrees today" % (salutation, name, temperature)
            celsius = (temperature - 32) * 5 / 9
            return (greeting, round(celsius, 2))
    

