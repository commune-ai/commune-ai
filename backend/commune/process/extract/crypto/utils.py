from copy import deepcopy
import os
import pandas as pd
import re
import ray
from commune.config import  ConfigLoader
from commune.client.minio.manager import MinioManager

import requests
@ray.remote
def run_query(query, url):
    # endpoint where you are making the request
    request = requests.post(url,
                            '',
                            json={'query': query.replace("'", '"')})
    if request.status_code == 200:
        request_json = request.json()
        if 'data' in request_json:
            return request_json['data']
        else:
            raise Exception('There was an error in the code fam {}', request_json)
    else:
        raise Exception('Query failed. return code is {}.      {}'.format(request.status_code, query))
