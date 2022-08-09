


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune.process.base import BaseProcess
from inspect import getfile
import socket
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
    api = GradioModule()

    find_registered_functions = GradioModule.find_registered_functions
    compile = GradioModule.compile
    register = GradioModule.register

    @staticmethod
    def run(self,host=None, port=7860, live=False, replace=True, **kwargs):

        port = port if port else GradioClient.api.suggest_port()
        host = host if host else GradioClient.api.host
        url = SimpleNamespace(**{ 'api': f'{host}:{GradioClient.api.port}',
                              'gradio' : f'{host}:{port}'})
        
        print('url', url.__dict__)

        allow_flagging = kwargs.get('flagging', 'never')

        gradio_metadata = { 
                           "host" : host, 
                           'url': url.gradio,
                           "live": live,
                           "port": port,
                        #    "module" : getfile(self.__class__), 
                        #    "name" : self.__class__.__name__,
                            **kwargs
                            }

        GradioClient.api.add_module(port=port, metadata=gradio_metadata)

        GradioClient.compile(self, live=live, allow_flagging=allow_flagging).launch(server_name= host ,server_port=port) 











if __name__== '__main__':
    import streamlit as st

    class TestModule(BaseProcess):
        cfg = dict(module='gradio.module.TestModule')

        def __init__(self, cfg=cfg):
            pass
        @GradioClient.register(inputs=["text", "text", gradio.Radio(["morning", "evening", "night"])], outputs="text")
        def Hello(self, Lname : str, Fname : str, day : 'list[any]'=["morning", "evening", "night"]) -> str:
            return "Hello, {} {}".format(Fname, Lname)  

        @GradioClient.register(inputs=["text", "text"], outputs="text")
        def goodbye(self, Fname : str, Lname : str) -> str:
            return "Goodbye, {} {}".format(Fname, Lname)  
        
        @GradioClient.register(inputs=["text", gradio.Checkbox() , gradio.Slider(0, 60)], outputs=["text", "number"])
        def greet(self, name, is_morning, temperature):
            salutation = "Good morning" if is_morning else "Good evening"
            greeting = "%s %s. It is %s degrees today" % (salutation, name, temperature)
            celsius = (temperature - 32) * 5 / 9
            return (greeting, round(celsius, 2))

        def run(self):
            GradioClient.run(self)
            
    # st.write(dir(TestModule))  
    import ray

    ray.shutdown()
    with ray.init(address='auto', namespace='commune'):
        test_instance = TestModule.deploy(actor=dict(refresh=False, name='test'))
        # st.write(GradioClient.run(test_instance))
        # st.write(ray.get(test_instance.run.remote()))

