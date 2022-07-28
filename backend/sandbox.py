import streamlit as st
import bittensor
import torch
graph = bittensor.metagraph().sync()
dataset = bittensor.dataset()
df = graph.to_dataframe()

st.write(graph.weights.shape)


