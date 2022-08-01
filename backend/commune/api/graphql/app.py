import os, sys
sys.path.append(os.environ["PWD"])
import datetime
import json
import pandas as pd
import streamlit as st
from commune.process import BaseProcess

config_loader = BaseProcess.config_loader
client = BaseProcess.default_clients(clients= ['graphql', 'mongo'])

st.write(client)
query_dict = {'module': 'config.manager.ConfigManager', 'fn': 'list_modules', 'override': {'actor.refresh': False}}
query_dict = json.loads(st.text_area("Input Json", json.dumps(query_dict)).replace("'", '"'))
query_list = client['mongo'].load(database='streamlit',  collection='historical_apis', query={})

st.write(query_list)
query_dict_str = json.dumps(query_dict).replace('"',"'")





if st.button("Query"):
    query = '''

    {{
        launch(input:"{json_string}")
    }}
    '''.format(json_string=query_dict_str)


    # query = '''

    # {{
    #     config(path: "client.mongo.manager")
    # }}
    # '''.format()

    st.write(query)
    st.write(json.loads(client['graphql'].query(query=query, return_one=True)))

    # client['mongo'].write(database='streamlit',  collection='historical_apis', data=query_dict)
