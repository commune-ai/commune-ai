from .utils import graphql_query
from commune.ray import ActorBase
import ray

class APIManager(ActorBase):
    default_cfg_path = f"{os.environ['PWD']}/commune/config/client/block/api.yaml"
    def __init__(
        self,
        cfg,
        # host= 'endpoints',
        # port= 8000
    ):
       
        self.url = f"http://{cfg['host']}:{cfg['port']}"

    def query(self,query, url=None, num_actors=2):

       if url != None:
            self.url = url
        if  isinstance(query, list):
            query_list = query
        else:
            query_list = [query]

        if len(query_list)== 1:
            return graphql_query(url=self.url, query=query)
        elif len(query_list)>1:
            ray_graphql_query = ray.remote(graphql_query)
            ready_jobs = [ray_graphql_query.remote(url=self.url, query=query)) for query in query_list]
            finished_jobs_results  = []
            while ready_jobs:
                ready_jobs, finished_jobs = ray.wait(ready_jobs)
                finished_jobs_results.extend(ray.get(finished_jobs))
            return finished_jobs_results

        else:
            raise f"Brooooooo, the query_list should be >= 1, but is actually {len(query_list)}"




