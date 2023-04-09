import pandas as pd
import datetime as dt
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import scipy.stats as stats

app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
route = '\premier_league\Datasets'
path = os.getcwd()
csv_files = glob.glob(os.path.join(path + route,"*.csv"))
dfs = []
seasons = []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)
    season = os.path.split(file)[-1].replace('.csv','')
    seasons.extend(season for i in range(len(df)))

data = pd.concat(dfs)
data['Season'] = seasons
data['Season'] = data['Season'].astype(str)
data.drop(data.iloc[:,28:82], axis=1,inplace=True)
data.replace(to_replace=['A','D','H'],value=['Away','Draw','Home'],inplace=True)
seasons = np.unique(np.array(seasons))

teams_data = pd.DataFrame(columns=['Season','Team','Home wins','Home draws','Home losses','Away wins','Away draws','Away losses','Full time home goals','Half time home goals','Full time away goals','Half time away goals','Home shots','Away shots','Home shots on target','Away shots on target','Home shots on woodwork','Away shots on woodwork','Home corners','Away corners','Home fouls','Away fouls','Home offsides','Away offsides','Home yellows','Away yellows','Home reds','Away reds'])

for season in seasons:
    season_data = data.loc[data['Season'] == season]
    teams = np.unique(np.array(season_data['HomeTeam']))
    season_array = np.array([season for i in range(len(teams))])
    for team in teams:
        home_data = season_data.loc[season_data['HomeTeam'] == team, ['FTHG','FTAG','HTHG','HS','HST','HHW','HC','HF','HO','HY','HR']]
        home_wins = home_data.loc[home_data['FTHG'] > home_data['FTAG']]
        home_losses = home_data.loc[home_data['FTHG'] < home_data['FTAG']]
        home_draws = home_data.loc[home_data['FTHG'] == home_data['FTAG']]
        away_data = season_data.loc[season_data['AwayTeam'] == team, ['FTHG','FTAG','HTAG','AS','AST','AHW','AC','AF','AO','AY','AR']]
        away_draws = away_data.loc[away_data['FTHG'] == away_data['FTAG']]
        away_wins = away_data.loc[away_data['FTHG'] < away_data['FTAG']]
        away_losses = away_data.loc[away_data['FTHG'] > away_data['FTAG']]
        team_data = np.array([season,team,len(home_wins),len(home_draws),len(home_losses),len(away_wins),len(away_draws),len(away_losses),home_data['FTHG'].sum(),home_data['HTHG'].sum(),away_data['FTAG'].sum(),away_data['HTAG'].sum(),home_data['HS'].sum(),away_data['AS'].sum(),home_data['HST'].sum(),away_data['AST'].sum(),home_data['HHW'].sum(),away_data['AHW'].sum(),home_data['HC'].sum(),away_data['AC'].sum(),home_data['HF'].sum(),away_data['AF'].sum(),home_data['HO'].sum(),away_data['AO'].sum(),home_data['HY'].sum(),away_data['AY'].sum(),home_data['HR'].sum(),away_data['AR'].sum()])
        teams_data.loc[len(teams_data)] = team_data

for column in teams_data.columns[2:]:
    teams_data[column] = teams_data[column].apply(pd.to_numeric)

ftr = data.groupby(by=['Season','FTR']).size().to_frame().reset_index().rename(columns={0:'Count'})
ftr_fig = px.histogram(ftr,x='Season',y='Count',color='FTR',barmode = 'group',height=600)
ftr_fig.update_layout(xaxis_type='category',yaxis_title='Count',legend_title_text="Full time result")

app.layout = dbc.Container(children=[
    html.H1('Premier league seasons analysis',className='p-6 mt-2'),
    dbc.Row(children = [
        dbc.Tabs(id = 'tabs', active_tab= 'goals', children=[
            dbc.Tab(tab_id = 'goals', label = 'Goals', children= [
                dbc.Row(children = [
                    html.H3('Goals per season',className='p-2'),
                    dbc.Col(md=2, children = [
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_2',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children = [
                        html.Div(children=[
                            dcc.Graph(id = 'goals_graph', figure={},config={'displayModeBar': False})
                        ])
                    ])
                ]),
                dbc.Row(children=[
                    html.H3('Team goals per season',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_1',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='team_goals_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ]),
                dbc.Row(children=[
                    html.H3('Goals distribution',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_11',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='goals_dist_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ])
            ]),
            dbc.Tab(tab_id = 'full_half_results', label='Results', children= [
                dbc.Row(children = [
                    html.H3('Full time results in each Premier League season',className='p-2'),
                    html.Div(children = [
                        dcc.Graph(figure=ftr_fig,config={'displayModeBar': False})
                    ]),
                ]),
                dbc.Row(children = [
                    html.H3('Team results per season',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_5',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='team_results_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ])
            ]),
            dbc.Tab(tab_id = 'fair_play', label='Fair Play', children= [
                dbc.Row(children=[
                    html.H3('Team cards per season',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_3',options=[{'label':x,'value':x} for x in seasons],multi=False,value='2000-01'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='team_cards_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ]),
                dbc.Row(children=[
                    html.H3('Cards and fouls correlation',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_4',options=[{'label':x,'value':x} for x in seasons],multi=False,value='2000-01'),
                    ]),
                    dbc.Col(md=5,children=[
                        html.Div(children = [
                            dcc.Graph(id='yellow_cards_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ]),
                    dbc.Col(md=5,children=[
                        html.Div(children = [
                            dcc.Graph(id='red_cards_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ]),
                dbc.Row(children = [
                    html.H3('Cards distribution',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_12',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='cards_dist_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ])
            ]),
            dbc.Tab(tab_id = 'shots', label='Shots', children=[
                dbc.Row(children=[
                    html.H3('Team shots per season',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_6',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='team_shots_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ]),
                dbc.Row(children=[
                    html.H3('Team shots on target per season',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_7',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='team_shots_ot_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ]),
                dbc.Row(children=[
                    html.H3('Shots correlation',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_8',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='shots_corr_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ]),
                    dbc.Row(children=[
                        html.H3('Shots distribution',className='p-2'),
                        dbc.Col(md=2,children=[
                            html.H4('Select season',className='p-3'),
                            dcc.Dropdown(id='season_slct_13',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                        ]),
                        dbc.Col(md=10,children=[
                            html.Div(children = [
                                dcc.Graph(id='shots_dist_graph', figure={},config={'displayModeBar': False})
                            ]),    
                        ])
                    ])
                ])
            ]),
            dbc.Tab(tab_id = 'corners', label = 'Corners', children=[
                dbc.Row(children=[
                    html.H3('Corners',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_9',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='corners_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ]),
                dbc.Row(children=[
                    html.H3('Corners correlation',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_10',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='corners_corr_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ]),
                dbc.Row(children=[
                    html.H3('Corners distribution',className='p-2'),
                    dbc.Col(md=2,children=[
                        html.H4('Select season',className='p-3'),
                        dcc.Dropdown(id='season_slct_14',options=[{'label':'All seasons','value':'All'}] + [{'label':x,'value':x} for x in seasons],multi=False,value='All'),
                    ]),
                    dbc.Col(md=10,children=[
                        html.Div(children = [
                            dcc.Graph(id='corners_dist_graph', figure={},config={'displayModeBar': False})
                        ]),    
                    ])
                ]),
            ]),
        ]),
    ]),
])

@app.callback(
    Output(component_id='team_goals_graph',component_property='figure'),
    Input(component_id='season_slct_1',component_property='value')
)

def update_team_goals_graph(season_slct_1):
    if season_slct_1 == 'All':
        selected_data = teams_data.groupby(by='Team')[['Full time home goals','Full time away goals']].sum().reset_index().sort_values(by='Team',ascending = False)
        fig_title = 'Goals in all seasons'
    else:
        selected_data = teams_data.loc[teams_data['Season'] == season_slct_1].sort_values(by='Team',ascending = False)
        fig_title = 'Goals in season ' + str(season_slct_1)
    fig = px.bar(selected_data,x=['Full time home goals','Full time away goals'],y='Team',orientation='h',height=800,title=fig_title,color_discrete_sequence=px.colors.sequential.Purples_r)
    fig.update_layout(yaxis_type='category',xaxis_title='Goals',yaxis_title='Team')
    return fig


@app.callback(
    Output(component_id='goals_graph',component_property='figure'),
    Input(component_id='season_slct_2',component_property='value')
)

def update_goals_graph(season_slct_2):
    if season_slct_2 == 'All':
        selected_data = teams_data[['Full time home goals','Full time away goals']].sum()
        selected_data['Total goals'] = selected_data['Full time home goals'] + selected_data['Full time away goals']
        fig_title = 'Goals in all seasons'
    else:
        selected_data = teams_data.groupby('Season').get_group(season_slct_2)[['Full time home goals','Full time away goals']].sum()
        selected_data['Total goals'] = selected_data['Full time home goals'] + selected_data['Full time away goals']
        fig_title = 'Goals in season' + str(season_slct_2)
    fig = px.pie(values=[selected_data['Full time home goals'],selected_data['Full time away goals']],names=['Full time home goals','Full time away goals'],hole=0.8,title=fig_title,color_discrete_sequence=px.colors.sequential.Purples_r)
    fig.update_traces(textposition='outside', textinfo='label+value')
    fig.update_layout(showlegend=False,annotations=[dict(text='Total goals: ' + str(selected_data['Total goals']), x=0.5, y=0.5, font_size=25, showarrow=False)])
    return fig

@app.callback(
    Output(component_id='team_cards_graph',component_property='figure'),
    Input(component_id='season_slct_3',component_property='value')
)

def update_team_cards_graph(season_slct_3):
    fair_play = teams_data.loc[teams_data['Season'] == season_slct_3][['Team','Home reds','Away reds','Home yellows','Away yellows']].sort_values(by='Team',ascending = False)
    fig = px.bar(fair_play,x=['Away reds','Home reds','Away yellows','Home yellows'],y='Team',orientation='h',height=600,title='Teams cards in season '+ season_slct_3,color_discrete_sequence=px.colors.sequential.Purples_r)
    fig.update_layout(yaxis_type='category',xaxis_title='Cards',yaxis_title='Team')
    return fig

@app.callback(
    [Output(component_id='yellow_cards_graph',component_property='figure'),
     Output(component_id='red_cards_graph',component_property='figure')],
    Input(component_id='season_slct_4',component_property='value')
)

def update_cards_graph(season_slct_4):
    fair_play = teams_data.loc[teams_data['Season'] == season_slct_4][['Home reds','Away reds','Home yellows','Away yellows','Home fouls','Away fouls']]
    fair_play['Total reds'] = fair_play['Home reds'] + fair_play['Away reds']
    fair_play['Total yellows'] = fair_play['Home yellows'] + fair_play['Away yellows']
    fair_play['Total fouls'] = fair_play['Home fouls'] + fair_play['Away fouls']
    fig_1 = px.scatter(fair_play,x='Total fouls',y='Total yellows',trendline='ols',title='Yellow cards and fouls correlation in '+ season_slct_4)
    fig_2 = px.scatter(fair_play,x='Total fouls',y='Total reds',trendline='ols',title='Red cards and fouls correlation in '+ season_slct_4)
    return fig_1,fig_2

@app.callback(
    Output(component_id='team_results_graph',component_property='figure'),
    Input(component_id='season_slct_5',component_property='value')
)

def update_team_results_graph(season_slct_5):
    if season_slct_5 == 'All':
        selected_data = teams_data.groupby(by='Team')[['Home wins','Home draws','Home losses','Away wins','Away draws','Away losses']].sum().reset_index().sort_values(by='Team',ascending=False)
        fig_title = 'Results in all seasons'
    else:
        selected_data = teams_data.groupby(by='Season').get_group(season_slct_5)[['Team','Home wins','Home draws','Home losses','Away wins','Away draws','Away losses']].sort_values(by='Team',ascending=False)
        fig_title = 'Results in season ' + season_slct_5
    fig = px.bar(selected_data,x=['Home wins','Away wins','Home draws','Away draws','Home losses','Away losses'],y='Team',barmode='stack',orientation='h',height=800,title=fig_title,color_discrete_sequence=px.colors.sequential.Purples_r)
    fig.update_layout(yaxis_type='category',xaxis_title='Results',yaxis_title='Team',legend_title_text="Games")
    return fig

@app.callback(
    Output(component_id='team_shots_graph',component_property='figure'),
    Input(component_id='season_slct_6',component_property='value')
)

def update_team_shots_graph(season_slct_6):
    if season_slct_6 == 'All':
        selected_data = teams_data.groupby(by='Team')[['Home shots','Away shots']].sum().reset_index().sort_values(by='Team',ascending=False)
        fig_title = 'Shots in all seasons'
    else:
        selected_data = teams_data.groupby(by='Season').get_group(season_slct_6)[['Team','Home shots','Away shots']].sort_values(by='Team',ascending=False)
        fig_title = 'Shots in season ' + season_slct_6
    fig = px.bar(selected_data,x=['Home shots','Away shots'],y='Team',barmode='stack',orientation='h',height=800,title=fig_title,color_discrete_sequence=px.colors.sequential.Purples_r)
    fig.update_layout(yaxis_type='category',xaxis_title='Shots',yaxis_title='Team',legend_title_text="Shots")
    return fig

@app.callback(
    Output(component_id='team_shots_ot_graph',component_property='figure'),
    Input(component_id='season_slct_7',component_property='value')
)

def update_team_shots_ot_graph(season_slct_7):
    if season_slct_7 == 'All':
        selected_data = teams_data.groupby(by='Team')[['Home shots on target','Away shots on target']].sum().reset_index().sort_values(by='Team',ascending=False)
        fig_title = 'Shots on target in all seasons'
    else:
        selected_data = teams_data.groupby(by='Season').get_group(season_slct_7)[['Team','Home shots on target','Away shots on target']].sort_values(by='Team',ascending=False)
        fig_title = 'Shots on target in season ' + season_slct_7
    fig = px.bar(selected_data,x=['Home shots on target','Away shots on target'],y='Team',barmode='stack',orientation='h',height=800,title=fig_title,color_discrete_sequence=px.colors.sequential.Purples_r)
    fig.update_layout(yaxis_type='category',xaxis_title='Shots on target',yaxis_title='Team',legend_title_text="Shots on target")
    return fig

@app.callback(
    Output(component_id='shots_corr_graph',component_property='figure'),
    Input(component_id='season_slct_8',component_property='value')
)

def update_shots_corr_graph(season_slct_8):
    if season_slct_8 == 'All':
        selected_data = teams_data.groupby(by='Team')[['Home shots on target','Away shots on target','Home shots','Away shots','Home wins','Away wins','Full time home goals','Full time away goals']].sum().reset_index().sort_values(by='Team',ascending=False)
        fig_title = 'Shots correlation with goals and wins'
    else:
        selected_data = teams_data.groupby(by='Season').get_group(season_slct_8)[['Team','Home shots on target','Away shots on target','Home shots','Away shots','Home wins','Away wins','Full time home goals','Full time away goals']].sort_values(by='Team',ascending=False)
        fig_title = 'Shots correlation with goals and wins in season ' + season_slct_8
    selected_data['Total shots'] = selected_data['Home shots'] + selected_data['Away shots']
    selected_data['Total shots on target'] = selected_data['Home shots on target'] + selected_data['Away shots on target']
    selected_data['Total wins'] = selected_data['Home wins'] + selected_data['Away wins']
    selected_data['Total goals'] = selected_data['Full time home goals'] + selected_data['Full time away goals']

    shots_wins = stats.linregress(x=selected_data['Total shots'],y=selected_data['Total wins'])
    target_wins = stats.linregress(x=selected_data['Total shots on target'],y=selected_data['Total wins'])
    shots_goals = stats.linregress(x=selected_data['Total shots'],y=selected_data['Total goals'])
    target_goals = stats.linregress(x=selected_data['Total shots on target'],y=selected_data['Total goals'])

    fig = make_subplots(rows=2,cols=2,subplot_titles=("Shots and wins correlation","Shots on target and wins correlation","Shots and goals correlation","Shots on target and goals correlation"))

    fig.add_trace(go.Scatter(x=selected_data['Total shots'],y=selected_data['Total wins'],mode='markers',text=selected_data['Team']),row=1,col=1)
    fig.add_trace(go.Scatter(x=selected_data['Total shots on target'],y=selected_data['Total wins'],mode='markers',text=selected_data['Team']),row=1,col=2)
    fig.add_trace(go.Scatter(x=selected_data['Total shots'],y=selected_data['Total goals'],mode='markers',text=selected_data['Team']),row=2,col=1)
    fig.add_trace(go.Scatter(x=selected_data['Total shots on target'],y=selected_data['Total goals'],mode='markers',text=selected_data['Team']),row=2,col=2)

    fig.add_trace(go.Scatter(x=selected_data['Total shots'],y=(shots_wins.intercept + shots_wins.slope*selected_data['Total shots']),mode='lines'),row=1,col=1)
    fig.add_trace(go.Scatter(x=selected_data['Total shots on target'],y=(target_wins.intercept + target_wins.slope*selected_data['Total shots on target']),mode='lines'),row=1,col=2)
    fig.add_trace(go.Scatter(x=selected_data['Total shots'],y=(shots_goals.intercept + shots_goals.slope*selected_data['Total shots']),mode='lines'),row=2,col=1)
    fig.add_trace(go.Scatter(x=selected_data['Total shots on target'],y=(target_goals.intercept + target_goals.slope*selected_data['Total shots on target']),mode='lines'),row=2,col=2)

    fig.update_xaxes(title_text="Shots", row=1, col=1)
    fig.update_xaxes(title_text="Shots on target", row=1, col=2)
    fig.update_xaxes(title_text="Shots", row=2, col=1)
    fig.update_xaxes(title_text="Shots on target", row=2, col=2)

    fig.update_yaxes(title_text="Wins", row=1, col=1)
    fig.update_yaxes(title_text="Wins", row=1, col=2)
    fig.update_yaxes(title_text="Goals", row=2, col=1)
    fig.update_yaxes(title_text="Goals", row=2, col=2)

    fig.update_layout(height=800,title_text = fig_title,showlegend = False)

    return fig

@app.callback(
    Output(component_id='corners_graph',component_property='figure'),
    Input(component_id='season_slct_9',component_property='value')
)

def update_corners_graph(season_slct_9):
    if season_slct_9 == 'All':
        selected_data = teams_data.groupby(by='Team')[['Home corners','Away corners']].sum().reset_index().sort_values(by='Team',ascending=False)
        fig_title = 'Corners in all seasons'
    else:
        selected_data = teams_data.groupby(by='Season').get_group(season_slct_9)[['Team','Home corners','Away corners']].sort_values(by='Team',ascending=False)
        fig_title = 'Corners in season ' + season_slct_9
    fig = px.bar(selected_data,x=['Home corners','Away corners'],y='Team',barmode='stack',orientation='h',height=800,title=fig_title,color_discrete_sequence=px.colors.sequential.Purples_r)
    fig.update_layout(yaxis_type='category',xaxis_title='Corners',yaxis_title='Team',legend_title_text="Corners")
    return fig

@app.callback(
    Output(component_id='corners_corr_graph',component_property='figure'),
    Input(component_id='season_slct_10',component_property='value')
)

def update_corners_corr_graph(season_slct_10):
    if season_slct_10 == 'All':
        selected_data = teams_data.groupby(by='Team')[['Home corners','Away corners','Home wins','Away wins','Full time home goals','Full time away goals']].sum().reset_index().sort_values(by='Team',ascending=False)
        fig_title = 'Corners correlation with goals and wins'
    else:
        selected_data = teams_data.groupby(by='Season').get_group(season_slct_10)[['Team','Home corners','Away corners','Home wins','Away wins','Full time home goals','Full time away goals']].sort_values(by='Team',ascending=False)
        fig_title = 'Corners correlation with goals and wins in season ' + season_slct_10
    selected_data['Total corners'] = selected_data['Home corners'] + selected_data['Away corners']
    selected_data['Total wins'] = selected_data['Home wins'] + selected_data['Away wins']
    selected_data['Total goals'] = selected_data['Full time home goals'] + selected_data['Full time away goals']

    corners_goals = stats.linregress(x=selected_data['Total corners'],y=selected_data['Total goals'])
    corners_wins = stats.linregress(x=selected_data['Total corners'],y=selected_data['Total wins'])

    fig = make_subplots(rows=1,cols=2,subplot_titles=("Corners and goals correlation","Corners and goals correlation"))

    fig.add_trace(go.Scatter(x=selected_data['Total corners'],y=selected_data['Total goals'],mode='markers',text=selected_data['Team']),row=1,col=1)
    fig.add_trace(go.Scatter(x=selected_data['Total corners'],y=selected_data['Total wins'],mode='markers',text=selected_data['Team']),row=1,col=2)

    fig.add_trace(go.Scatter(x=selected_data['Total corners'],y=(corners_goals.intercept + corners_goals.slope*selected_data['Total corners']),mode='lines'),row=1,col=1)
    fig.add_trace(go.Scatter(x=selected_data['Total corners'],y=(corners_wins.intercept + corners_wins.slope*selected_data['Total corners']),mode='lines'),row=1,col=2)

    fig.update_xaxes(title_text="Corners", row=1, col=1)
    fig.update_xaxes(title_text="Corners", row=1, col=2)

    fig.update_yaxes(title_text="Wins", row=1, col=1)
    fig.update_yaxes(title_text="Goals", row=1, col=2)

    fig.update_layout(height=600,title_text = fig_title,showlegend = False)

    return fig

@app.callback(
    Output(component_id='goals_dist_graph',component_property='figure'),
    Input(component_id='season_slct_11',component_property='value')
)

def update_goals_dist_graph(season_slct_11):
    if season_slct_11 == 'All':
        selected_data = data[['FTHG','FTAG']].rename(columns={'FTHG':'Home goals','FTAG':'Away goals'})
        fig_title = 'Goals distribution in all seasons'
    else:
        selected_data = data.groupby('Season').get_group(season_slct_11).rename(columns={'FTHG':'Home goals','FTAG':'Away goals'})
        fig_title = 'Goals distribution in season ' + season_slct_11

    fig = px.box(selected_data,y=['Home goals','Away goals'],title=fig_title)
    fig.update_yaxes(title_text="Count")
    fig.update_xaxes(title_text=" ")

    return fig

@app.callback(
    Output(component_id='cards_dist_graph',component_property='figure'),
    Input(component_id='season_slct_12',component_property='value')
)

def update_cards_dist_graph(season_slct_12):
    if season_slct_12 == 'All':
        selected_data = data[['HR','AR','HY','AY']].rename(columns={'HR':'Home reds','AR':'Away reds','HY':'Home yellows','AY':'Away yellows'})
        fig_title = 'Cards distribution in all seasons'
    else:
        selected_data = data.groupby('Season').get_group(season_slct_12).rename(columns={'HR':'Home reds','AR':'Away reds','HY':'Home yellows','AY':'Away yellows'})
        fig_title = 'Cards distribution in season ' + season_slct_12

    fig = px.box(selected_data,y=['Home reds','Away reds','Home yellows','Away yellows'],title=fig_title)
    fig.update_yaxes(title_text="Count")
    fig.update_xaxes(title_text=" ")

    return fig

@app.callback(
    Output(component_id='shots_dist_graph',component_property='figure'),
    Input(component_id='season_slct_13',component_property='value')
)

def update_cards_dist_graph(season_slct_13):
    if season_slct_13 == 'All':
        selected_data = data[['HS','AS','HST','AST']].rename(columns={'HS':'Home shots','AS':'Away shots','HST':'Home shots on target','AST':'Away shots on target'})
        fig_title = 'Shots distribution in all seasons'
    else:
        selected_data = data.groupby('Season').get_group(season_slct_13).rename(columns={'HS':'Home shots','AS':'Away shots','HST':'Home shots on target','AST':'Away shots on target'})
        fig_title = 'Shots distribution in season ' + season_slct_13

    fig = px.box(selected_data,y=['Home shots','Away shots','Home shots on target','Away shots on target'],title=fig_title)
    fig.update_yaxes(title_text="Count")
    fig.update_xaxes(title_text=" ")

    return fig

@app.callback(
    Output(component_id='corners_dist_graph',component_property='figure'),
    Input(component_id='season_slct_14',component_property='value')
)

def update_cards_dist_graph(season_slct_14):
    if season_slct_14 == 'All':
        selected_data = data[['HC','AC']].rename(columns={'HC':'Home corners','AC':'Away corners'})
        fig_title = 'Corners distribution in all seasons'
    else:
        selected_data = data.groupby('Season').get_group(season_slct_14).rename(columns={'HC':'Home corners','AC':'Away corners'})
        fig_title = 'Corners distribution in season ' + season_slct_14

    fig = px.box(selected_data,y=['Home corners','Away corners'],title=fig_title)
    fig.update_yaxes(title_text="Count")
    fig.update_xaxes(title_text=" ")

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)