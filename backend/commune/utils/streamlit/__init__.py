import streamlit as st

def describe(module =None, sidebar = True, detail=False, expand=True):
    
    _st = st.sidebar if sidebar else st
    st.sidebar.markdown('# '+str(module))
    fn_list = list(filter(lambda fn: callable(getattr(module,fn)) and '__' not in fn,  dir(module)))
    
    def content_fn(fn_list=fn_list):
        fn_list = _st.multiselect('fns', fn_list)
        for fn_key in fn_list:
            fn = getattr(module,fn_key)
            if callable(fn):
                _st.markdown('#### '+fn_key)
                _st.write(fn)
                _st.write(type(fn))
    if expand:
        with st.sidebar.expander(str(module)):
            content_fn()
    else:
        content_fn()



import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from mlflow.tracking import MlflowClient
from config import ConfigLoader
from experiment.utils import delete_experiment


def metric_distribution_over_trials(df, metric_column):
    fig = px.histogram(df,
                       x=metric_column,
                       title=f"{metric_column.upper()} over {len(df)} Trials")
    st.plotly_chart(fig)



class PlotModule:
    def __init__(self):


    def run(self, df: pd.DataFrame):
        with cols[1]:
            st.markdown("## Compare Hyperparameters Performance Across Experiments")
            plot_mode = st.radio("", ["scatter", "scatter2d", "Bar Plot", "Box Plot", "Heatmap 2D"], 1)


    def scatter2D(self, df=None):
        df = df if df else self.df
        column_options = list(df.columns)
        cols= st.columns([1,5])

        with cols[0]:
            st.markdown("## X Axis")
            x_col = st.selectbox("X Axis",column_options, 0 )

            st.markdown("## Y Axis")
            y_col = st.selectbox("Y Axis", column_options, 1)

            st.markdown("## Color Axis")
            color_col = st.selectbox("Color",  column_options + [None],  0)
            color_args = {"color": color_col} if color_col is not None else {}
            marker_size = st.slider("Select Marker Size", 5, 30, 20)


            df["size"] = [marker_size for _ in range(len(df))]
        with cols[1]:
            fig = px.scatter(df, x=x_col, y=y_col, size="size", **color_args)
            fig.update_layout(width=1000,
                            height=800)

            st.plotly_chart(fig)



    def scatter3D(self, df=None):
        with cols[0]:
            st.markdown("## X Axis")
            x_col = st.selectbox("X Axis", column_options, 0)
            st.markdown("## Y Axis")
            y_col = st.selectbox("Y Axis", column_options, 1)
            st.markdown("## Z Axis")
            z_col = st.selectbox("Z Axis", column_options, 2)

            st.markdown("## Color Axis")
            color_col = st.selectbox("Color", column_options + [None], 0)
            color_args = {"color": color_col} if color_col is not None else {}
            marker_size = st.slider("Select Marker Size", 5, 30, 20)
            df["size"] = [marker_size for _ in range(len(df))]


        with cols[1]:
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, size="size", **color_args)
            fig.update_layout(width=800, height=1000, font_size=15)
            st.plotly_chart(fig)  


    def box(self, df=None):
        with cols[0]:
            st.markdown("## X Axis")
            x_col = st.selectbox("X Axis", column_options, 0 )

            st.markdown("## Y Axis")
            y_col = st.selectbox("Y Axis", column_options, 0)


            st.markdown("## Color Axis")
            color_col = st.selectbox("Color", column_options + [None] ,  0)
            color_args = {"color":color_col} if color_col is not None else {}

            st.markdown("## Box Group Mode")
            boxmode = st.selectbox("Choose Box Mode", ["group", "overlay"], 0)

        with cols[1]:
            fig = px.box(df, x=x_col, y=y_col ,boxmode=boxmode, points=False, **color_args)
            fig.update_layout(width=1000, height=800, font_size=20)
            st.plotly_chart(fig)

    def bar(self):

        with cols[0]:

            st.markdown("## X Axis")
            x_col = st.selectbox("X Axis",column_options , 0 )

            st.markdown("## Y Axis")
            y_col = st.selectbox("Y Axis", column_options, 0)
            barmode = st.selectbox("Choose Bar Mode", ["relative", "group", "overlay"], 1)

            st.markdown("## Color Axis")
            color_col = st.selectbox("Color",  column_options + [None], 0 )
            color_args = {"color":color_col} if color_col is not None else {}

        with cols[1]:
            fig = px.bar(df, x=x_col, y=y_col,barmode=barmode, **color_args)

            fig.update_layout(width=1000, height=800, font_size=20)
            st.plotly_chart(fig)


    def headmap(self):
        # Choose X, Y and Color Axis
        with cols[0]:
            st.markdown("### X-axis")
            x_col = st.selectbox("Choose X-Axis Feature", column_options, 0)
            nbinsx = st.slider("Number of Bins", 10, 100, 10)

            st.markdown("### Y-axis")
            y_col = st.selectbox("Choose Y-Axis Feature", column_options, 0)
            nbinsy = st.slider("Number of Bins (Y-Axis)", 10, 100, 10)

            st.markdown("### Z-axis")
            z_col = st.selectbox("Choose Z-Axis Feature", column_options, 0)
            agg_func = st.selectbox("Aggregation Function", ["avg", "sum", "min", "sum", "count"], 0)

        with cols[1]:

            fig = px.density_heatmap(df,
                                x=x_col,
                                y=y_col,
                                z=z_col,
                                nbinsx=nbinsx,
                                nbinsy=nbinsy,
                                histfunc=agg_func)
            fig.update_layout(width=100, height=1000, font_size=20)
            st.plotly_chart(fig)



