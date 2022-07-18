import streamlit as st
from commune.client.ipfs.manager import IPFSManager
from commune.utils.misc import dict_put

client = IPFSManager.deploy()


fn_map = {}
for k in dir(client.client):
    if '__' not in k:
        fn_map[k] =       list(filter(lambda v: '__' not in v, dir(getattr(client.client,k))))
  


st.write(fn_map['files'])

st.write(client.client._client)
client.client.pubsub.publish('hello', 'message')

