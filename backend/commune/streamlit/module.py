

import os
import sys
sys.path[0] = os.environ['PWD']
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from mlflow.tracking import MlflowClient
from commune.plot.dag import DagModule 





class StreamlitPlotModule:

    height=1000
    width=1000
    theme= 'plotly_dark'
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


    def run(self, data, plots=[]):

        if len(plots) == 0:
            plots = self.plot_options


        supported_types = [pd.DataFrame]
        if isinstance(data, pd.DataFrame):
            with self.cols[1]:
                name2index = {_name:_idx for _idx, _name in enumerate(plots)}
                plot = 'st_'+st.selectbox('Choose a Plot', plots, name2index['histogram'])
            fig = getattr(self, plot)(df)
        else:
            raise NotImplementedError(f'Broooooo, hold on, you can only use the following {supported_types}')
        
        self.show(fig)
        
    @property
    def plot_options(self):
        plot_options = list(map(lambda fn: fn.replace('st_',''), self.streamlit_functions))
        return plot_options


    def show(self, fig):
        with self.cols[1]:
            st.plotly_chart(fig)

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

        
        fig = px.scatter(df, x=x_col, y=y_col, size="size", **color_args)
        fig.update_layout(width=1000,
                        height=800)

        return fig




    def st_scatter3D(self, df=None):
        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)

        plotly_kwargs = {}
        with self.cols[0]:
            st.markdown("## X Axis")
            plotly_kwargs['x'] = st.selectbox("X Axis", column_options, 0)
            st.markdown("## Y Axis")
            plotly_kwargs['y'] = st.selectbox("Y Axis", column_options, 1)
            st.markdown("## Z Axis")
            plotly_kwargs['z'] = st.selectbox("Z Axis", column_options, 2)
            st.markdown("## Color Axis")
            plotly_kwargs['color'] = st.selectbox("## Color", [None] + column_options, 0)
            marker_size = st.slider("Select Marker Size", 5, 30, 20)
            df["size"] = [marker_size for _ in range(len(df))]
            plotly_kwargs['size']= 'size'
            plotly_kwargs['template'] = self.theme

        fig = px.scatter_3d(df, **plotly_kwargs)
        fig.update_layout(width=self.width, height=self.height, font_size=15)
        return fig


    def st_box(self, df=None):


        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)
        plotly_kwargs = {}
        with self.cols[0]:
            st.markdown("## X Axis")
            plotly_kwargs['x'] = st.selectbox("X Axis", column_options, 0)
            st.markdown("## Y Axis")
            plotly_kwargs['y'] = st.selectbox("Y Axis", column_options, 1)
            st.markdown("## Color Axis")
            plotly_kwargs['color'] = st.selectbox("## Color", [None] + column_options, 0)
            marker_size = st.slider("Select Marker Size", 5, 30, 20)
            df["size"] = [marker_size for _ in range(len(df))]
            plotly_kwargs['template'] = self.theme
            st.markdown("## Box Group Mode")
            plotly_kwargs['boxmode'] = st.selectbox("Choose Box Mode", ["group", "overlay"], 0)

        fig = px.box(df, **plotly_kwargs)
        fig.update_layout(width=self.width, height=self.height, font_size=20)
        return fig

    def st_bar(self, df=None):

        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)


        plot_kwargs = {}
        with self.cols[0]:

            
            st.markdown("## X Axis")
            plot_kwargs['x'] = st.selectbox("X Axis",column_options , 0 )

            st.markdown("## Y Axis")
            plot_kwargs['y'] = st.selectbox("Y Axis", column_options, 0)
            plot_kwargs['barmode'] = st.selectbox("Choose Bar Mode", ["relative", "group", "overlay"], 1)

            st.markdown("## Color Axis")
            plot_kwargs['color'] = st.selectbox("Color",  [None] + column_options, 0 )

        fig = px.bar(df, **plot_kwargs)

        fig.update_layout(width=self.width, height=self.height, font_size=20)
        return fig




    def st_histogram(self, df=None):

        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)
        # Choose X, Y and Color Axis
        with self.cols[0]:
            plot_kwargs = {}
            st.markdown("### X-axis")
            plot_kwargs['x'] = st.selectbox("Choose X-Axis Feature", column_options, 0)
            plot_kwargs['nbins'] = st.slider("Number of Bins", 10, 100, 10)

            st.markdown("### Y-axis")
            plot_kwargs['y'] = st.selectbox("Choose Y-Axis Feature", column_options, 0)

            st.markdown("## Color Axis")
            plot_kwargs['color'] = st.selectbox("Color",  [None]+ column_options , 0 )
            # color_args = {"color":color_col} if color_col is not None else {}
            
            plot_kwargs['barmode'] = st.selectbox("Choose Bar Mode", ["relative", "group", "overlay"], 2)

        

        fig = px.histogram(df, **plot_kwargs)
        fig.update_layout(width=self.width, height=self.height, font_size=20)
        return fig


    def st_heatmap(self, df=None):

        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)
        # Choose X, Y and Color Axis

        plotly_kwargs = {}
        with self.cols[0]:
            st.markdown("### X-axis")
            plotly_kwargs['x'] = st.selectbox("Choose X-Axis Feature", column_options, 0)
            plotly_kwargs['nbinsx'] = st.slider("Number of Bins", 10, 100, 10)

            st.markdown("### Y-axis")
            plotly_kwargs['y'] = st.selectbox("Choose Y-Axis Feature", [None]+column_options, 0)
            plotly_kwargs['nbinsy'] = st.slider("Number of Bins (Y-Axis)", 10, 100, 10)

            st.markdown("### Z-axis")
            plotly_kwargs['z'] = st.selectbox("Choose Z-Axis Feature", column_options, 0)
            plotly_kwargs['histfunc'] = st.selectbox("Aggregation Function", ["avg", "sum", "min", "sum", "count"], 0)
            plotly_kwargs['template'] = self.theme

        fig = px.density_heatmap(df, **plotly_kwargs)
        fig.update_layout(width=self.width, height=self.height, font_size=20)

        return fig



if __name__ == '__main__':
    st.set_page_config(layout="wide")
    from sklearn.datasets import load_iris
    import pandas as pd
    st_plt = StreamlitPlotModule()
    data = load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)

    st_plt.run(data=df)
    # st.write(st_plt.streamlit_functions)


    
    # import json