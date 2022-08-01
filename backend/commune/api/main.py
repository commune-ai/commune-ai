

import os
import sys
import json
os.environ['PWD'] = os.getcwd()
sys.path.append(os.getcwd())
from fastapi import FastAPI
import ray
from ray import serve
from commune.api.launcher.module import Launcher

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
serve.start(detached=False)

@serve.deployment(route_prefix='/launcher')
@serve.ingress(app)
class API(Launcher):
  def __init__(self):

      self.count = 0
      self.launcher = Launcher.deploy(actor={'refresh':False})

#   @app.get("/")
#   def get(self):
#       return {"count": self.count}

  @app.post("/send")
  def send(self,module:str='process.bittensor.module.BitModule', fn:str='sync', args:list=[], kwargs:dict={}, override:dict={} ):


      if isinstance(args, str):
        args = json.loads(args)
      if isinstance(kwargs, str):
        kwargs = json.loads(args)


      job_id = ray.get(self.launcher.run_job.remote(module=module, fn=fn, args=args, kwargs=kwargs, override=override))
      return ray.get(job_id)


  @app.post("/actors")
  async def actors(self):
      
      return ray.get(self.launcher.getattr.remote('actor_names'))
      # print(ray.get(job_id))

  # @app.get("/decr")
  # def decr(self):
  #     self.count -= 1
  #     return {"count": self.count}


deployment = API.deploy()

api = API.get_handle()

print(ray.get(api.send.remote(module='process.bittensor.module.BitModule', fn='getattr', args= ['current_block'], override={'actor.refresh': False, 'network':'local'})))
print(ray.get((api.actors.remote())),'bro')
import requests
# requests.post('http://localhost:8000/launcher/send?fn=getattr&kwargs={"key":"n"}').json()