import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import dash
import datetime as dt
from matplotlib.colors import ListedColormap
from dash import dcc
from dash import html
from itertools import chain
from collections import Counter
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dfd = pd.read_csv('https://raw.githubusercontent.com/ZelshaR/Project/main/NoteBooks/CSVs/dfdd.csv')
dfd.index = dfd['Unnamed: 0'].values
del dfd['Unnamed: 0']
dfd['nu'] = 0
dfd['nu'] = dfd.index

dfc = pd.read_csv('https://raw.githubusercontent.com/ZelshaR/Project/main/NoteBooks/CSVs/dfcc.csv')
dfc.index = dfc['Unnamed: 0'].values
del dfc['Unnamed: 0']
dfc['nu'] = 0
dfc['nu'] = dfc.index

df = pd.read_csv('https://raw.githubusercontent.com/ZelshaR/Project/main/NoteBooks/CSVs/owid-covid-data.csv')

allcountli = list(dfc.columns)
allcountli.pop(0)
allcount = allcountli
for i in range(len(allcount)):
    allcount[i] = html.Option(value=allcount[i])

the_number_of_cases = dcc.RangeSlider(
    id='value-cases',
    min=0,
    max=557,
    value=[0, 557]
)
#это кореляция,её вычисление
lisc = list(dfc.columns)
inp='Afghanistan' #Afghanistan
inpcol=lisc.index(inp)
c = dfc.corr()
cd=c.iloc[:,[inpcol]]
cd=cd.sort_values(by=inp,ascending=False)
cd=cd[1:6]
sns.heatmap(cd,annot=True) # vmin=0,vmax=1
# plt.show()

corlist = dfc.columns #'это нармирование'
dfcsr = dfc.copy(deep=True)
dfdsr = dfd.copy(deep=True)
for i in corlist:
    for j in range(554):
        dfcsr[i].iloc[j] = (dfcsr[i].iloc[j]-min(dfcsr[i]))/(max(dfcsr[i])-min(dfcsr[i]))
for i in corlist:
    for j in range(554):
        dfdsr[i].iloc[j] = (dfdsr[i].iloc[j]-min(dfdsr[i]))/(max(dfdsr[i])-min(dfdsr[i]))

dfc.index = np.unique(df['date'])[150:704]
dfd.index = np.unique(df['date'])[150:704]

#'это превый грфик'
trace1 = go.Scatter(
        x=dfc.index,
        y=dfcsr['Afghanistan'],
        name='cases',
        yaxis='y2'
    )
trace2 = go.Scatter(
        x=dfd.index,
        y=dfdsr['Afghanistan'],
        name='death',
        yaxis='y2'
    )
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(trace1)
fig.add_trace(trace2, secondary_y=True)
fig['layout'].update(height=600, width=800, xaxis=dict(
        tickangle=45
    ))
fig.update_yaxes(range=[0, 1])
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='LightPink')
fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='blue', size=12))
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='LightPink')
# fig.show()

#'это второй грфик'
trace3 = go.Scatter(
        x=dfc.index,
        y=dfcsr['Afghanistan'],
        name='Russia',
        yaxis='y2'
    )
trace4 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[0].name],
        name=cd.iloc[0].name,
        yaxis='y2'
    )
trace5 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[1].name],
        name=cd.iloc[1].name,
        yaxis='y2'
    )
trace6 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[2].name],
        name=cd.iloc[2].name,
        yaxis='y2'
    )
trace7 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[3].name],
        name=cd.iloc[3].name,
        yaxis='y2'
    )
trace8 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[4].name],
        name=cd.iloc[4].name,
        yaxis='y2'
    )
fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(trace3)
fig3.add_trace(trace4)
fig3.add_trace(trace5)
fig3.add_trace(trace6)
fig3.add_trace(trace7)
fig3.add_trace(trace8)
fig3['layout'].update(height=600, width=800, xaxis=dict(
        tickangle=45
    ))
fig3.update_yaxes(range=[0, 1])
fig3.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='LightPink')
fig3.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='blue', size=12))
fig3.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='LightPink')
# fig3.show()

#это штука делает кореляцию не уродливой
fig2 = px.imshow(cd)
fig2.update_layout(width=500, height=600, margin=dict(l=200, r=200, b=100, t=100))

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(children=[
        html.H1(children='График роста и спада заболеваемости', style={
            'textAlign': 'center',
        }),

        dcc.Graph(id='num-of-cas',
                  figure=fig),
        html.Div(children='Срок, за который вы хотите видеть рост/спад заболеваемости (в месяцах)', style={
            'textAlign': 'center'}),
        html.Div(the_number_of_cases,
                 style={'width': '400px',
                        'margine-bottom': '40px'}),
        html.Div('Какю страну вы хотите увидеть?'),
        dcc.Input(
            id='txtinput',
            type='text',
            pattern=r"^[A-Za-z].*",
            list='browser',
            autoFocus=True,
            value='Russia'
        ),
    ], style={'padding': 10, 'flex': 1, "top": "50%",
              "left": "50%"}),

    html.Div(children=[
        html.H1(children='График стран с которыми наиболее коррелирует ', style={
            'textAlign': 'center',
        }),

        dcc.Graph(id='figure3',
                  figure=fig3)

    ]),

    html.Div(children=[
        dcc.Graph(id='cor',
                  figure=fig2)

    ], style={'padding': 10, 'flex': 1, "top": "50%",
              "left": "50%",
              'justify-content': 'center',
              'align-items': 'left', 'margin': 'auto'
              }),

    html.Datalist(id='browser', children=allcount),
    #     html.Div([
    #         dcc.Graph(id='num-of-cas1', figure=fig) #возможно
    #     ]),
], style={'alignitems': 'center', 'display': 'flex', 'flex-direction': 'row'})


@app.callback(
    Output(component_id='num-of-cas', component_property='figure'),
    #     Output(component_id='cor', component_property='figure'),
    Input(component_id='value-cases', component_property='value'),
    Input(component_id='txtinput', component_property='value'),
)
def update_num_of_cas(slider, con):
    w = dfcsr[slider[0]:slider[1]]
    ww = dfdsr[slider[0]:slider[1]]
    www = dfc[slider[0]:slider[1]]
    trace1 = go.Scatter(
        x=www.index,
        y=w[con],
        name='cases',
        yaxis='y2'
    )
    trace2 = go.Scatter(
        x=www.index,
        y=ww[con],
        name='death',
        yaxis='y2'
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(trace1)
    fig.add_trace(trace2, secondary_y=True)
    fig['layout'].update(height=600, width=800, xaxis=dict(
        tickangle=45
    ))
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='LightPink')
    fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='blue', size=12))
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='LightPink')

    return fig


@app.callback(
    Output(component_id='cor', component_property='figure'),
    Input(component_id='txtinput', component_property='value'),
)
def update_heatmap(con):
    lisc = list(dfc.columns)
    inp = con  # Afghanistan
    inpcol = lisc.index(inp)
    c = dfc.corr()
    cd = c.iloc[:, [inpcol]]
    cd = cd.sort_values(by=inp, ascending=False)
    cd = cd[1:6]
    sns.heatmap(cd, annot=True)  # vmin=0,vmax=1
    fig2 = px.imshow(cd)
    fig2.update_layout(width=500, height=600, margin=dict(l=200, r=200, b=100, t=100))

    return fig2


@app.callback(
    Output(component_id='figure3', component_property='figure'),
    Input(component_id='txtinput', component_property='value'),
)
def update_num_of_cas(con):
    lisc = list(dfc.columns)
    inp = con  # Afghanistan
    inpcol = lisc.index(inp)
    c = dfc.corr()
    cd = c.iloc[:, [inpcol]]
    cd = cd.sort_values(by=inp, ascending=False)
    cd = cd[1:6]

    trace3 = go.Scatter(
        x=dfc.index,
        y=dfcsr[con],
        name=con,
        yaxis='y2'
    )
    trace4 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[0].name],
        name=cd.iloc[0].name,
        yaxis='y2'
    )
    trace5 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[1].name],
        name=cd.iloc[1].name,
        yaxis='y2'
    )
    trace6 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[2].name],
        name=cd.iloc[2].name,
        yaxis='y2'
    )
    trace7 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[3].name],
        name=cd.iloc[3].name,
        yaxis='y2'
    )
    trace8 = go.Scatter(
        x=dfc.index,
        y=dfcsr[cd.iloc[4].name],
        name=cd.iloc[4].name,
        yaxis='y2'
    )
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(trace3)
    fig3.add_trace(trace4)
    fig3.add_trace(trace5)
    fig3.add_trace(trace6)
    fig3.add_trace(trace7)
    fig3.add_trace(trace8)
    fig3['layout'].update(height=600, width=800, xaxis=dict(
        tickangle=45
    ))
    fig3.update_yaxes(range=[0, 1])
    fig3.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='LightPink')
    fig3.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='blue', size=12))
    fig3.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='LightPink')

    return fig3

if __name__ == '__main__':
    app.run_server(use_reloader=False,debug=True)