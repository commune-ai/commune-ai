

import os
import sys

os.environ['PWD'] = os.getcwd()
sys.path.append(os.getcwd())
from fastapi import FastAPI
from starlette.graphql import GraphQLApp
from commune.api.graphql.query import schema
import datetime
import ray
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

app.add_route("/", GraphQLApp(schema=schema))