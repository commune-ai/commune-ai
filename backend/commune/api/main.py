

import os
import sys

os.environ['PWD'] = os.getcwd()
sys.path.append(os.getcwd())
from fastapi import FastAPI
import ray
from ray import serve
from commune.process.launcher.module import Launcher


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
class LauncherAPI(Launcher):
  def __init__(self):

      self.count = 0
      self.launcher = Launcher()

#   @app.get("/")
#   def get(self):
#       return {"count": self.count}

  @app.post("/send")
  def send(self,module:str, fn:str, kwargs:dict={}, override:dict={} ):
      print(x)

  # @app.get("/decr")
  # def decr(self):
  #     self.count -= 1
  #     return {"count": self.count}


deployment = API.deploy()

api = API.get_handle()
print(api.send.remote('BROOOOOOO'))