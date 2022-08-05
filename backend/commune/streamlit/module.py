

import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from mlflow.tracking import MlflowClient
from commune.plot.dag import DagModule 
class StreamlitPlotModule:
    def __init__(self):
        self.add_plot_tools()

        self.cols= st.columns([1,3])

        
    def add_plot_tools(self):
        # sync plots from express
        for fn_name in dir(px):
            if not (fn_name.startswith('__') and fn_name.endswith('__')):
                plt_obj = getattr(px, fn_name)
                if callable(plt_obj):
                    setattr(self, fn_name, plt_obj)

        self.dag = DagModule()

    @property
    def streamlit_functions(self):
        return [fn for fn in dir(self) if fn.startswith('st_')]  


    def run(self, data):
        if isinstance(data, pd.DataFrame):
            self.run_df(df=data)
        

    def run_df(self, df):
            
        with self.cols[1]:
            plot_options = list(map(lambda fn: fn.replace('st_',''), self.streamlit_functions))
            st.markdown("## Compare Hyperparameters Performance Across Experiments")
            # plot_mode = st.radio("",plot_options , 1)
            plot_mode = 'st_'+st.selectbox('Pick one', plot_options, 1)

        plot_fn = getattr(self, plot_mode)
        plot_fn(df)


    def st_scatter2D(self, df=None):
        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)


        with self.cols[0]:
            st.markdown("## X Axis")
            x_col = st.selectbox("X Axis",column_options, 0 )

            st.markdown("## Y Axis")
            y_col = st.selectbox("Y Axis", column_options, 1)

            st.markdown("## Color Axis")
            color_col = st.selectbox("Color",  column_options + [None],  0)
            color_args = {"color": color_col} if color_col is not None else {}
            marker_size = st.slider("Select Marker Size", 5, 30, 20)

            df["size"] = [marker_size for _ in range(len(df))]
        with self.cols[1]:
            fig = px.scatter(df, x=x_col, y=y_col, size="size", **color_args)
            fig.update_layout(width=1000,
                            height=800)

            st.plotly_chart(fig)




    def st_scatter3D(self, df=None):
        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)

        with self.cols[0]:
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


        with self.cols[1]:
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, size="size", **color_args)
            fig.update_layout(width=800, height=1000, font_size=15)
            st.plotly_chart(fig)  


    def st_box(self, df=None):


        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)
        with self.cols[0]:
            st.markdown("## X Axis")
            x_col = st.selectbox("X Axis", column_options, 0 )

            st.markdown("## Y Axis")
            y_col = st.selectbox("Y Axis", column_options, 0)


            st.markdown("## Color Axis")
            color_col = st.selectbox("Color", column_options + [None] ,  0)
            color_args = {"color":color_col} if color_col is not None else {}

            st.markdown("## Box Group Mode")
            boxmode = st.selectbox("Choose Box Mode", ["group", "overlay"], 0)

        with self.cols[1]:
            fig = px.box(df, x=x_col, y=y_col ,boxmode=boxmode, points=False, **color_args)
            fig.update_layout(width=1000, height=800, font_size=20)
            st.plotly_chart(fig)

    def st_bar(self, df=None):

        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)

        with self.cols[0]:

            st.markdown("## X Axis")
            x_col = st.selectbox("X Axis",column_options , 0 )

            st.markdown("## Y Axis")
            y_col = st.selectbox("Y Axis", column_options, 0)
            barmode = st.selectbox("Choose Bar Mode", ["relative", "group", "overlay"], 1)

            st.markdown("## Color Axis")
            color_col = st.selectbox("Color",  column_options + [None], 0 )
            color_args = {"color":color_col} if color_col is not None else {}

        with self.cols[1]:
            fig = px.bar(df, x=x_col, y=y_col,barmode=barmode, **color_args)

            fig.update_layout(width=1000, height=800, font_size=20)
            st.plotly_chart(fig)





    def st_heatmap(self, df=None):

        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)
        # Choose X, Y and Color Axis
        with self.cols[0]:
            st.markdown("### X-axis")
            x_col = st.selectbox("Choose X-Axis Feature", column_options, 0)
            nbinsx = st.slider("Number of Bins", 10, 100, 10)

            st.markdown("### Y-axis")
            y_col = st.selectbox("Choose Y-Axis Feature", column_options, 0)
            nbinsy = st.slider("Number of Bins (Y-Axis)", 10, 100, 10)

            st.markdown("### Z-axis")
            z_col = st.selectbox("Choose Z-Axis Feature", column_options, 0)
            agg_func = st.selectbox("Aggregation Function", ["avg", "sum", "min", "sum", "count"], 0)

        with self.cols[1]:

            fig = px.density_heatmap(df,
                                x=x_col,
                                y=y_col,
                                z=z_col,
                                nbinsx=nbinsx,
                                nbinsy=nbinsy,
                                histfunc=agg_func)
            fig.update_layout(width=100, height=1000, font_size=20)
            st.plotly_chart(fig)




if __name__ == '__main__':

    from sklearn.datasets import load_iris
    import pandas as pd
    st_plt = StreamlitPlotModule()
    data = load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)

    st_plt.run(data=df)
    st.write(st_plt.streamlit_functions)


    
    # import json
