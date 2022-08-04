

import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from mlflow.tracking import MlflowClient
import datetime
from streamlit_agraph import agraph, Node, Edge, Config




                    # **kwargs e.g. node_size=1000 or node_color="blue"
                    

class d3Dag:
    nodes = []
    edges = []
    graph = []

    Node = Node 
    Edge = Edge

    cfg = dict(
            width=500, 
            height=500, 
            directed=True,
            nodeHighlightBehavior=True, 
            highlightColor="#F7A7A6", # or "blue"
            collapsible=True,
            node={'labelProperty':'label'},
            link={'labelProperty': 'label', 'renderLabel': True},
            update_threshold = 60
        )

    last_update_timestamp:int = datetime.datetime.utcnow().timestamp()
    def __init__(self,  cfg=None):
        if isinstance(cfg, dict):
            self.cfg.update(cfg)

        self.update_threshold = self.cfg['update_threshold']

    @property
    def time_since_last_update(self):
        return datetime.datetime.utcnow().timestamp()- self.last_update_timestamp

    def add_node(self, node=None, id:str=None, size:int=None, update=False ,**kwargs):

        
        if isinstance(node, Node):
            node = node
        elif isinstance(node, dict):
            node = Node(**node)
        elif node is None:
            node = Node(id=id, size=size, **kwargs)

        self.nodes.append( node)

        if update:
            self.run()
    def add_edge(self, edge=None, source:str=None, target:str=None, label:str=None, type:str='CURVE_SMOOTH', update=False,  **kwargs):
        

        if isinstance(edge, Edge):
            edge = edge
        elif isinstance(edge, dict):
            edge = Edge(**edge)

        elif edge is None:
            edge = Edge(source=source, 
                                    label=label, 
                                    target=target, 
                                    type=type)

        self.edges.append(edge)

        if update:
            self.run()

    def add_nodes(self,nodes:list, update=True):
        for node in nodes:
            node['update'] = False
            self.add_node(node)
    
        if update:
            self.run()
        
    def add_edges(self,edges:list, update=True):
        for edge in edges:
            self.add_edge(edge)
        
        if update:
            self.run()
    
    def run(self,nodes:list=None, edges:list=[]):

        
        if isinstance(nodes, list):
            self.add_nodes(nodes=nodes, update=False)
        if isinstance(edges,list):
            self.add_edges(edges=edges, update=False)


        self.graph = agraph(nodes=self.nodes, 
                edges=self.edges, 
                config=Config(**self.cfg))




        self.last_update_timesamp = datetime.datetime.utcnow().timestamp()

        return self.graph

    def build(self,*args, **kwargs):
        return  self.run(*args, **kwargs)
    def render(self,*args, **kwargs):
        return  self.run(*args, **kwargs)


    def add_graph(self,*args, **kwargs):
        return  self.run(*args, **kwargs)

    


if __name__ == '__main__':
    import streamlit as st


    nodes = [
        dict(id="Spiderman", 
                    label="Peter Parker", 
                    color='blue',
                    size=600),

        dict(id="Superman", 
                    label="Peter Parker", 
                    color='green',
                    size=600),
        dict(id="Captain_Marvel", 
                    size=400, 
                    svg="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_captainmarvel.png") 
                    
    ]


    edges = [dict(source="Captain_Marvel", target="Spiderman"), dict(source="Superman", target="Spiderman")]



    dag = d3Dag()

    dag.add_graph(nodes=nodes, edges=edges)

    dag.add_node(node=dict(id="Bro", 
                    label="Peter Parker", 
                    color='blue',
                    size=600), update=False)


    
    # import json
