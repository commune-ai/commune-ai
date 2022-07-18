

import os
import sys

os.environ['PWD'] = os.getcwd()
sys.path.append(os.getcwd())
from fastapi import FastAPI
import ray
from ray import serve


from fastapi.middleware.cors import CORSMiddleware

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

app = FastAPI()

# app.add_middleware(
#   CORSMiddleware,
#   # allow_origins=["http://localhost:3000"],
#   allow_origins=["*"],
#   allow_credentials=True,
#   allow_methods=["*"],
#   allow_headers=["*"]
# )

ctx = ray.init(address='auto', namespace='api')


@serve.deployment
@serve.ingress(app)
class Counter:
  def __init__(self):
      self.count = 0

  @app.get("/")
  def get(self):
      return {"count": self.count}

  @app.get("/incr")
  def incr(self):
      self.count += 1
      return {"count": self.count}

  @app.get("/decr")
  def decr(self):
      self.count -= 1
      return {"count": self.count}

serve.start(detached=True)
Counter.deploy()

