import os, sys
sys.path.append(os.environ["PWD"])
import datetime
import json
import pandas as pd
import streamlit as st
from commune.client.manager import ClientManager
    
client_manager = ClientManager.deploy(actor=False)
client = client_manager.client
query_dict = {'module': 'config.manager.ConfigManager', 
              'fn': 'run', 
              'kwargs': {'query': {'module': 'data.regression.crypto.sushiswap.dataset'}} }

query_dict = json.loads(st.text_area("Input Json", json.dumps(query_dict)))


if st.button("Query"):
    query = '''

    {{
        launch(input:"{json_string}")
    }}
    '''.format(json_string=json.dumps(query_dict).replace('"',"'"))


    st.write(query)
    st.write(json.loads(client['graphql'].query(query=query, return_one=True)))