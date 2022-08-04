


import os, sys
sys.path.append(os.environ['PWD'])




from commune.process.base import BaseProcess


class GradioAPI(BaseProcess):
    default_cfg_path =  'gradio.api.module'
    def __init__(self, cfg=None):
        BaseProcess.__init__(self, cfg=cfg)

        self.port2module = {} 
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

    def add_module(self, port, json:dict):
        self.modules['active'][port]
        visable.append(current)
        return jsonify({"executed" : True})

    def rm_module(self, ):
        current = request.json
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
        for fn_key in GradioAPI.get_funcs(self):
            if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioAPI.register.__name__:
                fn_keys.append(fn_key)
        return fn_keys


    @staticmethod
    def get_funcs(self):
        return [func for func in dir(self) if not func.startswith("__") and callable(getattr(self, func, None)) ]


    @staticmethod
    def has_registered_functions(self):
        '''
        find the registered functions
        '''
        for fn_key in GradioAPI.get_funcs(self):
            if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioAPI.register.__name__:
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


        st.write(module_list)

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
    def has_gradio(self):
        return GradioAPI.has_registered_functions(self)



    def suggest_port(self, max_trial_count=10):
        trial_count = 0 
        for port in range(*self.port_range):
            if not self.portConnection(port):
                return port

        '''
        TODO: kill a port when they are all full
        '''
        raise Exception(f'There does not exist an open port between {self.port_range}')
        
    @staticmethod
    def compile(self, live=False, flagging='never', theme='default', **kwargs):
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


if __name__ == '__main__':
    from importlib import import_module
    import streamlit as st
    import inspect
    st.write(__name__)
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)


    def filter_gradio_classes(self):
        module_classes = [c for c in clsmembers if c[1].default_cfg_path]

    api = GradioAPI()
    # st.write(api.list_modules())
    st.write(api)