import dash
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np


def plot_parallel_coord_interactive(names, stdnames, means, std, col):
    names = ["x1", "x2", "x3"]
    stdnames = [name + "_std" for name in names]
    means = pd.DataFrame(np.random.rand(10, 3), columns=names)
    std = pd.DataFrame(np.random.rand(10, 3) * 0.1, columns=stdnames)

    df = means.join(std)
    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            dash_table.DataTable(
                id="datatable-interactivity",
                columns=[
                    {"name": i, "id": i, "deletable": True, "selectable": True}
                    for i in df.columns
                ],
                data=df.to_dict("records"),
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                row_deletable=True,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=10,
            ),
            html.Div(id="datatable-interactivity-container"),
        ]
    )


    @app.callback(
        Output("datatable-interactivity", "style_data_conditional"),
        [Input("datatable-interactivity", "selected_columns")],
    )
    def update_styles(selected_columns):
        return [
            {"if": {"column_id": i}, "background_color": "#D2F3FF"}
            for i in selected_columns
        ]


    @app.callback(
        Output("datatable-interactivity-container", "children"),
        [
            Input("datatable-interactivity", "derived_virtual_data"),
            Input("datatable-interactivity", "derived_virtual_selected_rows"),
        ],
    )
    def update_graphs(rows, derived_virtual_selected_rows):
        # When the table is first rendered, `derived_virtual_data` and
        # `derived_virtual_selected_rows` will be `None`. This is due to an
        # idiosyncracy in Dash (unsupplied properties are always None and Dash
        # calls the dependent callbacks when the component is first rendered).
        # So, if `rows` is `None`, then the component was just rendered
        # and its value will be the same as the component's dataframe.
        # Instead of setting `None` in here, you could also set
        # `derived_virtual_data=df.to_rows('dict')` when you initialize
        # the component.
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows = []

        dff = df if rows is None else pd.DataFrame(rows)

        colors = [
            "#7FDBFF" if i in derived_virtual_selected_rows else "#A9A9A9"
            for i in range(len(dff))
        ]
        fig = go.Figure()
        for i in range(len(dff)):
            fig.add_scatter(
                x=[1, 2, 3],
                y=dff[names].loc[i].values,
                error_y=dict(
                    type="data",  # value of error bar given in data coordinates
                    array=dff[stdnames].loc[i].values,
                    visible=True,
                ),
                line = dict(color=colors[i]),
            )

        return [dcc.Graph(id='plot', figure=fig)]


    if __name__ == "__main__":
        app.run_server(debug=True)
