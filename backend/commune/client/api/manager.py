import os
import sys
import json
os.environ['PWD'] = os.getcwd()
sys.path.append(os.getcwd())
from commune.ray import ActorBase
import ray
import requests

class APIManager(ActorBase):
    default_cfg_path = f"client.api.manager"
    def __init__(
        self,
        cfg,
        # host= 'endpoints',
        # port= 8000
    ):
        self.host = cfg['host']
        self._url = f"http://{cfg['host']}:{cfg['port']}"

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, url):
        self._url = url

    def get(self, url:str=None,endpoint:str=None, params={},**kwargs):
        if url is None:
            url = self.url

        if endpoint:
            url = os.path.join(url, endpoint)
            
        return requests.get(url=url, params=params, **kwargs).json()

    def post(self, url:str=None, endpoint:str=None, params:dict={},**kwargs):
        if url is None:
            url = self.url
            
        if endpoint:
            url = os.path.join(url, endpoint)
        return requests.post(url=url, params=params, **kwargs).json()

if __name__ == '__main__':
    api = APIManager.deploy(actor=False)
    # print(api.get(endpoint='launcher/send', 
    #                  params=dict(module='process.bittensor.module.BitModule', fn='getattr', kwargs='{"key": "n"}')))

    print(api.get(endpoint='module_tree'))


