


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune.process.base import BaseProcess
from inspect import getfile
import socket
from typing import *

from commune.utils.misc import SimpleNamespace
import streamlit as st
st.write('FUCK')

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
    default_cfg_path =  'gradio.api.module'


    def __init__(self, cfg=None):
        BaseProcess.__init__(self, cfg=cfg)

        self.port2module = {} 
        self.module2port = {}
        self.host  = self.cfg.get('host', '0.0.0.0')
        self.port  = self.cfg.get('port', 8000)
        self.num_ports = self.cfg.get('num_ports', 10)
        self.port_range = self.cfg.get('port_range', [7860, 7865])
        

    @property
    def active_modules(self):
        return self._modules

    @property
    def gradio_modules(self):
        return self._modules

    def add_module(self, port, metadata:dict):
        self.port2module[port] = metadata
        # self.module2port[module]
        return True

    def rm_module(self, port):
        print(current)
        visable.remove(current)
        return jsonify({"executed" : True,
                        "ports" : current['port']})


    @staticmethod
    def find_registered_functions(self):
        '''
        find the registered functions
        '''
        fn_keys = []

        for fn_key in GradioModule.get_funcs(self):
            try:
                if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioModule.register.__name__:
                    fn_keys.append(fn_key)
            except:
                continue
        return fn_keys


    @staticmethod
    def get_funcs(self):

        fn_list =  [func for func in self.functions()]
        return fn_list


    @staticmethod
    def has_registered_functions(self):
        '''
        find the registered functions
        '''
        for fn_key in GradioModule.get_funcs(self):
            if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioModule.register.__name__:
                return True


        return False



    def list_modules(self, mode='config'):

        assert mode in ['config', 'module']

        '''
        mode: options are active (running modules) all and inactive
        '''

        module_config_list = list(map( lambda x: x['config'], self.config_manager.module_tree(tree=False)))


        module_list = []
        for m_cfg_path in module_config_list:

            try:
                m_cfg = self.config_loader.load(m_cfg_path)

                object_module = self.get_object(m_cfg['module'])
                if self.has_gradio(object_module):
                    module_list.append(object_module)
            except:
                continue


        return module_list


    def active_port(self, port):
        is_active = port in self.port2module
        return is_active


    def portConnection(self ,port : int):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        result = s.connect_ex((self.host, port))
        if result == 0: return True
        return False


    @staticmethod
    def has_gradio(module):

        if isinstance(module, dict):
            module_cfg = module
        else:
            module_cfg =  getattr(module, 'cfg', {})

        gradio_cfg = module_cfg.get('gradio')
        if gradio_cfg == None:
            return False
        elif type(gradio_cfg) == bool:
            return gradio_cfg
        elif type(gradio_cfg) in [dict, list, tuple]:
            return len(gradio_cfg)


    def suggest_port(self, max_trial_count=10):
        trial_count = 0 
        for port in range(*self.port_range):
            print(port, 'port', not self.portConnection(port))
            if not self.portConnection(port):
                return port

        '''
        TODO: kill a port when they are all full
        '''
        raise Exception(f'There does not exist an open port between {self.port_range}')
        
    @staticmethod
    def compile(self, live=False, flagging='never', theme='default', **kwargs):
        print("Just putting on the finishing touches... 🔧🧰")
        
        for func in GradioModule.find_registered_functions(self):
            registered_fn = getattr(self, func)
            registered_fn()


        demos = []
        names = []
        for fn_key, fn_params in self.registered_gradio_functons.items():                
            names.append(func)
            demos.append(gradio.Interface(fn=getattr(self, fn_key),
                                        inputs=fn_params['inputs'],
                                        outputs=fn_params['outputs'],
                                        theme='default'))
            print(f"{fn_key}....{bcolor.BOLD}{bcolor.OKGREEN} done {bcolor.ENDC}")


        print("\nHappy Visualizing... 🚀")
        return gradio.TabbedInterface(demos, names)


    @staticmethod
    def launch_gradio(self,host='0.0.0.0', port=7860, live=False, replace=True, **kwargs):
        url = SimpleNamespace(**{ 'api': f'{host}:{port}',
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

        GradioModule.compile(self, live=live, allow_flagging=allow_flagging).launch(server_name= host ,server_port=port) 


    def run_module(self, module, **kwargs):
        module = self.get_object(module)
        if self.has_gradio(module):
            gradio_kwargs = dict(host=self.host, port=self.suggest_port(), live=False, replace=True)
            # module.cfg['gradio'] = gradio_kwargs
            
        return module

    def register_fn(self, module,  inputs:Union[dict, list], outputs: Union[dict, list]):
        return self.get_object(module)

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

    # without '__reduce__', the instance is unserializable.
    def __reduce__(self):
        deserializer = GradioModule
        serialized_data = (self.cfg,)
        return deserializer, serialized_data



class TestModule(BaseProcess):
    cfg = dict(module='gradio.module.TestModule', 
                        gradio=True)

    # def __init__(self, cfg=cfg):
    @GradioModule.register(inputs=["text", "text", gradio.Radio(["morning", "evening", "night"])], outputs="text")
    def Hello(self, Lname : str, Fname : str, day : 'list[any]'=["morning", "evening", "night"]) -> str:
        return "Hello, {} {}".format(Fname, Lname)  

    @GradioModule.register(inputs=["text", "text"], outputs="text")
    def goodbye(self, Fname : str, Lname : str) -> str:
        return "Goodbye, {} {}".format(Fname, Lname)  
    
    @GradioModule.register(inputs=["text", gradio.Checkbox() , gradio.Slider(0, 60)], outputs=["text", "number"])
    def greet(self, name, is_morning, temperature):
        salutation = "Good morning" if is_morning else "Good evening"
        greeting = "%s %s. It is %s degrees today" % (salutation, name, temperature)
        celsius = (temperature - 32) * 5 / 9
        return (greeting, round(celsius, 2))


    def launch_gradio(self):
        GradioModule.launch_gradio(self)






# st.write(gr_module) 
import ray
ray.shutdown()
with ray.init(address='auto', namespace='commune'):
    gr_module = GradioModule.deploy(actor=False)
    
    
    _TestModule = gr_module.run_module(module='gradio.api.module.TestModule')

    # st.write(GradioModule.get_funcs(test_module))
    # ray.get(test_module.launch_gradio.remote())
    test_instance = TestModule.deploy(actor=dict(refresh=True, name='test'))
    # # st.write(GradioClient.run(test_instance))
    
    st.write(ray.get(test_instance.launch_gradio.remote()))

