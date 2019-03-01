# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 21:59:27 2018
Nested Dropdown https://github.com/plotly/dash-recipes/blob/master/dash-three-cascading-dropdowns.py
@author: Kevin Zhang
"""
#https://dash.plot.ly/installation
#pip install dash==0.22.0  # The core dash backend
#pip install dash-renderer==0.13.0  # The dash front-end
#pip install dash-html-components==0.11.0  # HTML components
#pip install dash-core-components==0.26.0  # Supercharged components
#pip install plotly --upgrade  # Plotly graphing library used in examples
from datetime import timedelta
import pandas as pd
import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go


def HierarchyLoader(Hierarchy = 's3://elasticbeanstalk-us-east-2-882532749654/DashData/Hierarchy.xlsx'):
    CurrentYear = pd.ExcelFile(Hierarchy)
    today = pd.to_datetime(CurrentYear.parse(0).RefreshDate[0])
    Hierarchy = CurrentYear.parse('Hierarchy', header = 4)
    
    FoodPrice = pd.ExcelFile(r's3://elasticbeanstalk-us-east-2-882532749654/DashData/08 - August - 2018 Canada Foods Metric Price List.XLSX')
    FoodPrice = FoodPrice.parse(0)[['Case UPC/GTIN', 'Brkt 1 Price','Selling Units']]

    HPCPrice = pd.ExcelFile(r's3://elasticbeanstalk-us-east-2-882532749654/DashData/08 - August - 2018 Canada HPC Price List Metric.XLSX')
    HPCPrice = HPCPrice.parse(0)[['Case UPC/GTIN', 'Brkt 1 Price','Selling Units']]

    SVGPrice = pd.ExcelFile(r's3://elasticbeanstalk-us-east-2-882532749654/DashData/04-April-2018 SVG Price List Metric.XLSX')
    SVGRetail = SVGPrice.parse(0)[['Case UPC/GTIN', 'Brkt 1 Price','Selling Units']]
    SVGECOMM = SVGPrice.parse(1)[['Case UPC/GTIN', 'Brkt 1 Price','Selling Units']]    
    
    CasePrice = pd.concat([FoodPrice,HPCPrice,SVGRetail,SVGECOMM], ignore_index = True)
    CasePrice.columns = ['CaseUPC','PricePerCase','UnitPerCase']
    CasePrice.CaseUPC.replace(regex = True, inplace = True, 
                          to_replace = r'\D',value = r'') # remove any non digit values
    CasePrice.CaseUPC = CasePrice.CaseUPC.str[1:-1] # remove first and last character, Case UPC is 12 digits 
    
    Hierarchy['CaseUPC'] = Hierarchy.UPC.str[:12]
    Hierarchy = Hierarchy.merge(CasePrice, on = 'CaseUPC', how = 'left')
     
    return today, Hierarchy

today, Hierarchy = HierarchyLoader()

def generate_table(dataframe, max_rows=10):
    return html.Table(
            className = 'AccuracyTable',
            children = [
                    html.Thead(
                            children = [html.Tr([html.Th(col) for col in dataframe.columns])]
                            ),
                    html.Tbody(
                            children = [html.Tr([
                                            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                                                ]) for i in range(min(len(dataframe), max_rows))]
                            )
                    
                        ]
                    )

Prediction = pd.read_csv("s3://elasticbeanstalk-us-east-2-882532749654/DashData/HairOutputv4.csv", index_col = [0])
#PredictivePower = pd.read_csv("HairPredictivePowerv4.csv")

Prediction = Prediction.reset_index().merge(Hierarchy, on = ['UPC', 'Planned Account'], how = 'left')
DataRobot = pd.read_csv("s3://elasticbeanstalk-us-east-2-882532749654/DashData/MoneyballRobot.csv")
Prediction = Prediction.merge(DataRobot, on = ['Date', 'UPC', 'Planned Account'], how = 'left')

Prediction['Date'] = pd.to_datetime(Prediction['Date'])


app = dash.Dash(__name__)
application = app.server

app.layout = html.Div([
    
    html.Img([
            
            ]),
    
    html.Div([ # Create a checkbox for Predicting and non-predicting
            dcc.RadioItems(
                id='Prediction-checkbox',
                options = [
                    {'label': 'Predicting', 'value': 'Pre'},
                    {'label': 'Everything', 'value': 'Eve'}
                ],
                value = ['Pre']
            )
            ]),

    html.Div([
            
        html.Div([
                
            dcc.Dropdown( # Create a dropdown for Customer Hierarchy
            id='CustH-dropdown',
            options=[{'label': k, 'value': k} for k in Hierarchy.columns[5:9].values],
            value= Hierarchy.columns[4:7].values[2]),
                ],style={'width': '48%', 'display': 'inline-block'}),
    
            
                    
        html.Div([
                  
            dcc.Dropdown( # Create a dropdown for Product Hierarchy
            id='ProdH-dropdown',
            options=[{'label': k, 'value': k} for k in Hierarchy.columns[0:5].values],
            value= Hierarchy.columns[0:4].values[0]),
                ],style={'width': '48%', 'display': 'inline-block'}),
                    
        ], style={"height" : "70%", "width" : "100%"}),
        
        html.Hr(),
        
        html.Div([
            html.Div([
                      
                dcc.Dropdown( # Create a Product Dropdown, reading from id = ProdH-dropdown
                id='Customer-dropdown',
                ),
                    ],style={'width': '48%', 'display': 'inline-block'}),
            
            
            html.Div([
                    
                dcc.Dropdown( # Create a Customer Dropdown, reading from id = CustH-dropdown
                id='Product-dropdown',
                ),
                    ],style={'width': '48%', 'display': 'inline-block'}),
        
                ], style={"height" : "70%", "width" : "100%"}),
        html.Div([
                dcc.Graph(id='Prediction_plot')
        ]),
        
        html.Div([
                html.H4(children='Model Accuracy'), 
                html.Table(id='Accuracy-table')], 
            style={'width': '60%','display': 'inline-block', 'padding': '0 20','vertical-align': 'middle'})

    ])
                        

@app.callback(
    dash.dependencies.Output('Customer-dropdown', 'options'),
    [dash.dependencies.Input('Prediction-checkbox', 'value'),
     dash.dependencies.Input('CustH-dropdown', 'value')])
def set_Customer_options(Predicting, selected_CustH):
    if Predicting == 'Pre':
        Pre = Prediction[Prediction['Predicting'] == True]       
    else:
        Pre = Prediction
    return [{'label': i, 'value': i} for i in Pre[selected_CustH].unique()]
@app.callback(
    dash.dependencies.Output('Customer-dropdown', 'value'))
def set_Customer_value(available_Cust):
    return available_Cust[0]

@app.callback(
    dash.dependencies.Output('Product-dropdown', 'options'),
    [dash.dependencies.Input('Prediction-checkbox', 'value'),
     dash.dependencies.Input('ProdH-dropdown', 'value'),
     dash.dependencies.Input('CustH-dropdown', 'value'),
     dash.dependencies.Input('Customer-dropdown', 'value')])
def set_Product_options(Predicting, selected_ProdH, selected_CustH, selected_Customer):
    if Predicting == 'Pre':
        Pre = Prediction[Prediction['Predicting'] == True]       
    else:
        Pre = Prediction
    # subset Hierachy with associated customer
    Pre = Pre[Pre[selected_CustH] == selected_Customer]
    return [{'label': i, 'value': i} for i in Pre[selected_ProdH].unique()]
@app.callback(
    dash.dependencies.Output('Product-dropdown', 'value'))
def set_Product_value(available_Prod):
    return available_Prod[0]


@app.callback(
    dash.dependencies.Output('Prediction_plot', 'figure'),
    [dash.dependencies.Input('Prediction-checkbox', 'value'),
     dash.dependencies.Input('ProdH-dropdown', 'value'),
     dash.dependencies.Input('CustH-dropdown', 'value'),
     dash.dependencies.Input('Customer-dropdown', 'value'),
     dash.dependencies.Input('Product-dropdown', 'value')])
def set_display_children(Predicting, selected_ProdH, selected_CustH, selected_Customer, selected_Product):
    if Predicting == 'Pre':
        Pre = Prediction[Prediction['Predicting'] == True]       
    else:
        Pre = Prediction

    target = Pre[(Pre[selected_CustH] == selected_Customer) & (Pre[selected_ProdH] == selected_Product)].groupby(['Date']).sum()
    target['Cases'] = target['Cases'].loc[:today]
    target['Internal Plan'] = target['Internal Plan'].loc[today:pd.to_datetime('2018-12-31')]

    return {
            'data': [
                    go.Scatter(
                    y = target['Prediction'],
                    x = target['Prediction'].index,
                    name = 'Prediction',
                    line = dict(
                            color = ('#b5b203'),
                            width = 3,
                            dash = 'dot'),
                    connectgaps=True),
                                     
                    go.Scatter(
                    y = target['Cases'],
                    x = target['Cases'].index,
                    name = 'Actual',
                    line = dict(
                        color = ('rgb(55, 128, 191)'),
                        width = 2)),
                            
                    go.Scatter(
                    y = target['Internal Plan'],
                    x = target['Internal Plan'].index,
                    name = 'Internal Plan',
                    line = dict(
                            color = ('black'),
                            width = 2,
                            dash = 'dash')),
                            
                    go.Scatter(
                    y = target['DataRobot'],
                    x = target['DataRobot'].index,
                    name = 'DataRobot',
                    line = dict(
                            color = ('grey'),
                            width = 2,
                            dash = 'dash'))
                    ],
            'layout': [
                    go.Layout(
                    title = 'Customer:{} <br> Product:{}<br>'.\
                    format(selected_Customer, selected_Product),
                    titlefont=dict(
                            family='Arial',
                            size=18,
                            color='#08110b'),
    
                    paper_bgcolor='rgb(255,255,255)',
                    plot_bgcolor='rgb(229,229,229)',
                    xaxis=dict(
                        gridcolor='rgb(255,255,255)',
                        showgrid=True,
                        showline=False,
                        showticklabels=True,
                        tickcolor='rgb(127,127,127)',
                        ticks='outside',
                        zeroline=False
                    ),
                    yaxis=dict(
                                gridcolor='rgb(255,255,255)',
                                showgrid=True,
                                showline=False,
                                tickcolor='rgb(127,127,127)',
                                ticks='outside',
                                zeroline=False
                            ),
                        )
                    ]
            }

@app.callback(
    dash.dependencies.Output('Accuracy-table', 'children'),
    [dash.dependencies.Input('Prediction-checkbox', 'value'),
     dash.dependencies.Input('ProdH-dropdown', 'value'),
     dash.dependencies.Input('CustH-dropdown', 'value'),
     dash.dependencies.Input('Customer-dropdown', 'value'),
     dash.dependencies.Input('Product-dropdown', 'value')])
def Accuracytable_update(Predicting, selected_ProdH, selected_CustH, selected_Customer, selected_Product):
    if Predicting == 'Pre':
        Pre = Prediction[Prediction['Predicting'] == True]       
    else:
        Pre = Prediction

    target = Pre[(Pre[selected_CustH] == selected_Customer) & (Pre[selected_ProdH] == selected_Product)].groupby(['Date']).sum()
    target['Cases'] = target['Cases'].loc[:today]
    target['Internal Plan'] = target['Internal Plan'].loc[today:pd.to_datetime('2018-12-31')]
    
    MonthAcc = target.loc[today - timedelta(weeks = 16):today] # Each 4 Months Accuracy
    MonthlyAccuracy = pd.DataFrame() # All month Accuracy
    WeeklyAccuracy = []
    Week, Month = 1., 1.

    for i in range(MonthAcc.shape[0]): # Loop for 4 month
        
        if MonthAcc.Cases.iloc[i] > 0:
            Week  = abs(MonthAcc.Cases.iloc[i] - MonthAcc.Prediction.iloc[i])/MonthAcc.Cases.iloc[i]
        else:
            pass
        # if accuracy is >1 we count is as inaccurate 0
        if Week > 1:
            Week = 1
        WeeklyAccuracy = np.append(WeeklyAccuracy, 1-Week)
    
    for i in range(3): # Loop for 4 month, validate on 3 months

        if MonthAcc.Cases[MonthAcc.index.month == (today.month - i - 1)].sum() > 0:
            Month = abs(MonthAcc.Cases[MonthAcc.index.month == (today.month - i - 1)].sum() - MonthAcc.Prediction[MonthAcc.index.month == (today.month - i - 1)].sum())/MonthAcc.Cases[MonthAcc.index.month == (today.month - i - 1)].sum()
        else:
            Month = np.nan
        # if accuracy is >1 we count is as inaccurate 0
        if Month > 1:
            Month = 1
        MonthlyAccuracy['{}'.format((today - pd.DateOffset(months= i + 1)).strftime("%B"))] = [1-Month]

    MonthlyAccuracy['Monthly Accuracy'] = np.nanmean(MonthlyAccuracy)
    MonthlyAccuracy['Weekly Accuracy'] = np.nanmean(WeeklyAccuracy)
    MonthlyAccuracy['Internal Plan YTG'] = target['Internal Plan'].loc[today:'2018-12-31'].sum()
    MonthlyAccuracy['Prediction YTG'] = target['Prediction'].loc[today:'2018-12-31'].sum()
    MonthlyAccuracy['2017 Same Period YTG'] = target['Cases'].loc[today - timedelta(days = 365):'2017-12-31'].sum()
    MonthlyAccuracy['2018 YTD Actual with Internal Plan'] = target['Cases'].loc['2018-01-01':today].sum()+target['Internal Plan'].loc[today:'2018-12-31'].sum()
    MonthlyAccuracy['2018 YTD Actual with Prediction'] = target['Cases'].loc['2018-01-01':today].sum()+target['Prediction'].loc[today:'2018-12-31'].sum()
    MonthlyAccuracy['2018 YTD Prediction Only'] = target['Prediction'].loc['2018-01-01':'2018-12-31'].sum()
    
    return generate_table(MonthlyAccuracy.round(3))

if __name__ == '__main__':
    application.run(host="0.0.0.0")