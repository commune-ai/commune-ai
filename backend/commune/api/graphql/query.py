import os
import sys
import datetime
import graphene
import ray
from commune.utils.misc import dict_put, dict_has
import pandas as pd
import numpy as np
import json
from commune.api.graphql.manager import QueryModule
# experiment_manager = ExperimentManager.initialize(spawn_ray_actor=False, actor_name='experiment_manager')
# context = ray.init(address='auto', namespace='graphql')
query_module = QueryModule.deploy(actor=False)
# inference_manager = InferenceManager.deploy(actor=False)
class Query(graphene.ObjectType):
    alltokens = graphene.Field(graphene.List(graphene.String),
                              mode=graphene.String(default_value="commune_app"))

    launch = graphene.JSONString(input=graphene.String(required=True))
    def resolve_launch(root, info, input):
        input_dict = json.loads(input.replace("'", '"'))
        output_dict = query_module.launch(job_kwargs=input_dict, block=True)
        return output_dict
    config =  graphene.JSONString(path=graphene.String(required=True))
    def resolve_config(parent, info, **kwargs):
        return query_module.get_config(path=kwargs['path'])

    moduleTree =  graphene.JSONString(root=graphene.String(required=False, default_value='/app/commune'))
    def resolve_moduleTree(parent, info, **kwargs):
        out_dict = query_module.get_module_tree(root=kwargs['root'])

        return out_dict

# inference_manager = InferenceManager.initialize()
schema = graphene.Schema(query=Query)
    