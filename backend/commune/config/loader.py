import os
import re
import sys
if __name__== "__main__":
    sys.path.append(os.environ['PWD'])
import yaml
from copy import deepcopy
# sys.path.append(os.environ['PWD'])
from commune.utils.misc import dict_get,dict_put, list2str
from functools import partial
import glob

class ConfigLoader:
    """
    Loads Modular and Customizable Configs

    Available Tricks
        - use ENV variablesx
        - pull in any component of other configs
        - link

    """

    cache= {}
    cfg = None
    cnt = 0
    root =  os.environ['PWD']+'/commune'

    def __init__(self, path=None,
                 local_var_dict = {},
                 load_config=True):
        '''
        path: path to yaml config
        local_var_dict: variables you want to replace with ENV variables

        '''
        
        if load_config:
            self.load(path=path, local_var_dict=local_var_dict)


    @staticmethod
    def override_cfg(cfg, override={}):
        """
        
        """

        for k,v in override.items():
            dict_put(input_dict=cfg,keys=k, value=v)

        return cfg



    def load(self, path, local_var_dict={}, override={}, parse_only=False):
        self.local_var_dict = local_var_dict
        self.cfg = self.parse_config(path=path)
       
        if not parse_only:
            self.cfg = self.resolver_methods(cfg=self.cfg)
            self.cfg = self.override_cfg(cfg=self.cfg, override=override)
        return self.cfg

    # def get_base_cfg(self, cfg,  key_path, local_key_path=[]):
    #     if isinstance(cfg, str):
    #         config_path = re.compile('^(get_base_cfg)\((.+)\)').search(input)

    #         # if there are any matches ()
    #         if config_path:
    #             config_path = config_path.group(2)

    
    def resolve_config_path(self, config_path):
        # find config path

        if '.yaml' in config_path:
            config_path = config_path.replace('.yaml', '')
        config_path = config_path.replace(".", "/")

        if self.root != config_path[:len(self.root)]:
            config_path =  os.path.join(self.root,config_path)
        
        if os.path.isdir(config_path):
            file_path_list = glob.glob(f'{config_path}/*')
        else:
            file_path_list = glob.glob(f'{config_path}.*')
        

        found_config_path_list = []
        for file_path in file_path_list:
            if '.' in file_path and '.yaml' in file_path:
                found_config_path_list += [file_path]
                break

        assert len(found_config_path_list) == 1, f'BRO {len(found_config_path_list)} {config_path}'
        config_path = found_config_path_list[0] 

        return config_path
    def get_cfg(self, input, key_path, local_key_path=[]):
        
        """

        :param
            input: input string (str)
        :return:
             Regex Match
                - path of config within match
             No Regex Match
                - None (this means its not pointing to a config path)
        """

        cfg=input

        if isinstance(cfg, str):
            config_path = re.compile('^(get_cfg)\((.+)\)').search(input)

            # if there are any matches ()
            if config_path:
                config_path = config_path.group(2)
                config_keys =  None
                if ',' in config_path:
                    assert len(config_path.split(',')) == 2
                    config_path ,config_keys = config_path.split(',')

                cfg = self.parse_config(config_path)
                cfg = self.resolve_config(cfg=cfg,root_key_path=key_path, local_key_path=key_path)

                if config_keys != None:

                    cfg =  dict_get(input_dict=cfg, keys=config_keys)

        return cfg

    def set_cache(self, key, value):
        self.cache[key] = value
    

    def get_cache(self, key):
        return self.cache[key]
        
        

    def local_copy(self, input, key_path):
        """

        :param
            input: input string (str)
        :return:
             Regex Match
                - path of config within match
             No Regex Match
                - None (this means its not pointing to a config path)
        """

        variable_object = input


        if isinstance(input, str):

            variable_path = re.compile('^(local_copy)\((.+)\)').search(input)

            if variable_path:
                variable_path = variable_path.group(2)

                # get the object
                local_cfg_key_path = self.cache[list2str(key_path)]
                
                if local_cfg_key_path:
                    local_cfg = dict_get(input_dict=self.cfg, keys=self.cache[list2str(key_path)])
                else: 
                    local_cfg = self.cfg
                variable_object = dict_get(input_dict=local_cfg,
                                                    keys = variable_path)

        return variable_object


    def copy(self, input, key_path):
        """

        :param
            input: input string (str)
        :return:
             Regex Match
                - path of config within match
             No Regex Match
                - None (this means its not pointing to a config path)
        """

        variable_object = input


        if isinstance(input, str):

            variable_path = re.compile('^(copy)\((.+)\)').search(input)

            if variable_path:
                variable_path = variable_path.group(2)

                # get the object
                try:
                    variable_object = dict_get(input_dict=self.cfg,
                                                        keys = variable_path)
                except KeyError as e:
                    raise(e)

        
        return variable_object


    def get_inner_variable(self, input, key_path):
        pattern = re.compile('.*?\${(\w+)}.*?')



        if isinstance(input, str):
            match = pattern.findall(input)  # to find all env variables in line

            for g in match:
                full_value = input.replace(
                    f'${{{g}}}', self.resolve_variable(cfg=input, root_key_path= key_path)
                )
        return full_value
    def get_variable(self, input, key_path):

        output = self.copy(input=input, key_path=key_path)
        output = self.local_copy(input=input, key_path=key_path)
        return output

    def resolve_variable(self, cfg, root_key_path = []):
        '''
        :return:
        '''

        if isinstance(cfg, dict):
            keys = list(cfg.keys())
        elif isinstance(cfg, list):
            keys = list(range(len(cfg)))
        
        for k in keys:
            key_path = root_key_path +[k]
            cfg[k] = self.get_variable(input=cfg[k], key_path=key_path )
            if type( cfg[k]) in [list, dict]:
                cfg[k] = self.resolve_variable(cfg=cfg[k], root_key_path= key_path)

        return cfg

    def resolve_config(self, cfg=None, root_key_path=[], local_key_path=[]):

        if isinstance(cfg, dict):
            keys = list(cfg.keys())
        elif isinstance(cfg, list):
            keys = list(range(len(cfg)))
        else:
            return cfg
        for k in keys:
            key_path = root_key_path + [k]

            # registers the current key path in the local config path (for local referencing)
            key_path_str = list2str(key_path)
            if key_path_str not in self.cache:
                self.cache[key_path_str] = local_key_path

            cfg[k] = self.get_cfg(input=cfg[k],
                                  key_path= key_path,
                                  local_key_path=local_key_path)

            if type(cfg[k]) in [list, dict]:
                cfg[k] = self.resolve_config(cfg=cfg[k],
                                  root_key_path= key_path,
                                  local_key_path=local_key_path)

        return cfg

    def resolver_methods(self, cfg):
        '''
        :param path: path to config
        :return:
            config
        '''
        self.cfg = cfg

        # composes multiple config files
        self.cfg = self.resolve_config(cfg=self.cfg)

        # fills in variables (from ENV as well as local variables)
        self.cfg = self.resolve_variable(cfg=self.cfg)

        return self.cfg

    def parse_config(self,
                     path=None,
                     tag='!ENV'):

        if type(path) in [dict, list]:
            return path

        assert isinstance(path, str), path

        local_var_dict = self.local_var_dict
        
        path = self.resolve_config_path(path)

        """
        Load a yaml configuration file and resolve any environment variables
        The environment variables must have !ENV before them and be in this format
        to be parsed: ${VAR_NAME}.
        E.g.:
            client:
                host: !ENV ${HOST}
                port: !ENV ${PORT}
            app:
                log_path: !ENV '/var/${LOG_PATH}'
                something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'

        :param
            str path: the path to the yaml file
            str tag: the tag to look for

        :return
            dict the dict configuration
        """
        # pattern for global vars: look for ${word}
        pattern = re.compile('.*?\${(\w+)}.*?')
        loader = yaml.SafeLoader

        # the tag will be used to mark where to start searching for the pattern
        # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
        loader.add_implicit_resolver(tag, pattern, None)

        def constructor_env_variables(loader, node):
            """
            Extracts the environment variable from the node's value
            :param yaml.Loader loader: the yaml loader
            :param node: the current node in the yaml
            :return: the parsed string that contains the value of the environment
            variable
            """

            
            value = loader.construct_scalar(node)
            match = pattern.findall(value)  # to find all env variables in line
            if match:
                full_value = value
                for g in match:
                    full_value = full_value.replace(
                        f'${{{g}}}', os.environ.get(g, self.local_var_dict[g] if g in self.local_var_dict else None)
                    )
                return full_value
            return value

        loader.add_constructor(tag,constructor_env_variables)
        with open(path) as conf_data:
            cfg =  yaml.load(conf_data, Loader=loader)
        
        
        return cfg
if __name__== "__main__":
    print(ConfigLoader(path=path).cfg) 

