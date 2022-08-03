import os, sys
sys.path.append(os.environ["PWD"])

import requests
import gradio as gr
import socket
import random
from inspect import getfile

from commune.process.base import BaseProcess

DOCKER_LOCAL_HOST = 'localhost'

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


    


class GradioModule(BaseProcess):
    default_cfg_path = 'gradio.module'
    
    active_port_map = {}
    port_count = 0
    def __init__(self, cfg=None):
        BaseProcess.__init__(self, cfg=cfg)

        self.num_ports = self.cfg.get('num_ports', 10)
        self.port_range = self.cfg.get('port_range', [7860, 7865])



    def get_funcs(self):
        return [func for func in dir(self.cls) if not func.startswith("__") and callable(getattr(self.cls, func, None)) ]

    def compile(self, live=False, allow_flagging='never', theme='default', **kwargs):
        print("Just putting on the finishing touches... 🔧🧰")
        for func in self.get_funcs():
            this = getattr(self.cls, func, None)
            if this.__name__ == "wrap":
                this()

        demos = []
        names = []


        interface_dict = {}

        for fn_key, param in self.get_registered_gradio_functons().items():                
            names.append(fn_key)
            interface_dict[fn_key] = gr.Interface(fn=getattr(self.cls, fn_key, None),
                                                    inputs=param['inputs'],
                                                    outputs=param['outputs'],
                                                    live=live,
                                                    allow_flagging=allow_flagging,
                                                    theme=theme)
            print(f"{func}....{bcolor.BOLD}{bcolor.OKGREEN} done {bcolor.ENDC}")

        print("\nHappy Visualizing... 🚀")
        return gr.TabbedInterface(interface_dict.values(), interface_dict.keys())
        
    def get_registered_gradio_functons(self):
        try:
            self.cls.registered_gradio_functons
        except AttributeError:
            return None
        return self.cls.registered_gradio_functons
    

    def run(self,  **kwargs):
        server_port= kwargs["port"] if "port" in kwargs else self.determinePort() 
        api_port =  kwargs[ 'listen' ] if 'listen' in kwargs else '5000'
        api_url = f'http://{DOCKER_LOCAL_HOST}:{api_port}'
        gradio_url = f'http://{DOCKER_LOCAL_HOST}:{port}'
        live = kwargs.get('live', False)
        allow_flagging = kwargs.get('flagging', 'never')

        gradio_metadata = {"port" : server_port, "host" : gradio_url, "file" : getfile(self.cls.__class__), "name" : self.cls.__class__.__name__, "kwargs" : kwargs}
        try:
            self.client['api'].post(endpoint=, json=gradio_metadata)
        except Exception:
            print(f"**{bcolor.BOLD}{bcolor.FAIL}CONNECTION ERROR{bcolor.ENDC}** 🐛The listening api is either not up or you choose the wrong port.🐛")
            return
        try:
            requests.post(f"{api_url}/api/remove/port", json=gradio_metadata)
        except Exception:
            print(f"**{bcolor.BOLD}{bcolor.FAIL}CONNECTION ERROR{bcolor.ENDC}** 🐛The api either lost connection or was turned off...🐛")
        
        
        self.compile(live=live, allow_flagging=allow_flagging).launch(server_port=server_port) 

    def portConnection(self ,port : int):
        s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
                
        result = s.connect_ex((DOCKER_LOCAL_HOST, port))
        if result == 0: return True
        return False

    def active_port(self, port:int):
        return self.active_port_map.get(port, False)
    
    def determinePort(self, max_trial_count=10):
        trial_count = 0 
        for port in range(*self.port_range):
            if not self.portConnection(port):
                return port
        raise Exception(f'There does not exist an open port between {self.port_range}')
        
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
            return wrap
        return register_gradio


if __name__== '__main__':
    import streamlit as st

    gr = GradioModule()
    st.write(gr.client['api'].get(endpoint=''))
    
