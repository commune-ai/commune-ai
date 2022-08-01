

import os
import sys
import json
os.environ['PWD'] = os.getcwd()
sys.path.append(os.getcwd())
from fastapi import FastAPI
import ray
from ray import serve
from commune.process.launcher.module import Launcher
from commune.api.module import APIBase


from fastapi.middleware.cors import CORSMiddleware

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  # allow_origins=["http://localhost:3000"],
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

ray.init(address='auto', namespace='serve')
serve.start(detached=True)

@serve.deployment(route_prefix='/')
@serve.ingress(app)
class API(APIBase):
  def __init__(self):
      APIBase.__init__(self)
      # self.launcher = Launcher.deploy(actor={'refresh':True})
#   @app.get("/")
#   def get(self):
#       return {"count": self.count}
  @app.get("/launch")
  async def launch(self,module:str='process.bittensor.module.BitModule', fn:str='sync', args:str='[]', kwargs:str='{}', override:dict={} ):
    return APIBase.launch(self=self, module=module, fn=fn, args=args, kwargs=kwargs)  

  @app.get("/actors")
  async def actors(self):
    return APIBase.actors(self)

  @app.get("/module_tree")
  async def module_tree(self):
    return APIBase.module_tree(self)



deployment = API.deploy()

api = API.get_handle()

import requests

print(requests.get('http://localhost:8000/module_tree', params= dict(module='process.bittensor.module.BitModule', fn='getattr', args='["n"]')).text)
