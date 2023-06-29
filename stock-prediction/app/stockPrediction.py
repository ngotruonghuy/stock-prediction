# Import xây dựng giao diện Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
#Import để tạo biểu đồ
import plotly.graph_objs as go
#Import để xử lý callback trong Dash
from dash.dependencies import Input, Output, State
#Import để chuẩn hóa dữ liệu
from sklearn.preprocessing import MinMaxScaler
#Import để làm việc với dữ liệu dạng bảng
import pandas as pd
import numpy as np
import math

# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# from keras.models import load_model
# from keras.layers import Dropout

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.layers import Dropout

import xgboost
import pickle

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

#Tạo 1 ứng dụng dash bằng Dash framework
app = dash.Dash()
server = app.server

#Tạo 1 đối tượng TimeSeries từ Alpha Vantage API bằng cách cung cấp khóa API ('key') và định dạng dầu ra
ts = TimeSeries(key='D8JHWTNSXO7M9VKV', output_format='pandas')
ti = TechIndicators(key='D8JHWTNSXO7M9VKV', output_format='pandas')

#Lấy dữ liệu giao dịch và chỉ số kỹ thuật từ Alpha Vantage cho công ty được chỉ định và lưu nó vào CSV
def update_data(companyName):

    #lấy dữ liệu giao dịch
    data = ts.get_intraday(symbol=companyName,interval='15min', outputsize='full') # data
    data = data[0]
    data.rename(lambda x: x[2:].strip(), axis='columns', inplace=True)

    #Lấy các chỉ số kĩ thuật từ Alpha Vantage
    indicator_roc = ti.get_roc(symbol=companyName, interval='15min', time_period=20) # roc
    indicator_roc = indicator_roc[0]
    indicator_sma = ti.get_sma(symbol=companyName, interval='15min', time_period=20) # sma
    indicator_sma = indicator_sma[0]
    indicator_rsi = ti.get_rsi(symbol=companyName, interval='15min', time_period=20) # rsi
    indicator_rsi = indicator_rsi[0]
    indicator_bb = ti.get_bbands(symbol=companyName, interval='15min', time_period=20) # bbands
    indicator_bb = indicator_bb[0]

    # Sử dụng phương thức merge để hợp nhất các khung dữ liệu thành 1 khung dữ liệu gốc (original_df)
    original_df = pd.merge(data, indicator_roc, on='date', how='inner')
    original_df = pd.merge(original_df, indicator_sma, on='date', how='inner')
    original_df = pd.merge(original_df, indicator_rsi, on='date', how='inner')
    original_df = pd.merge(original_df, indicator_bb, on='date', how='inner')

    #Đảo ngược thứ tự các hàng trong khung dữ liệu gốc (original_df)
    original_df = original_df.iloc[::-1]
    #Lưu vào file csv
    original_df.to_csv('../data/' + companyName +'.csv')

# Nhận danh sách các chỉ số kỹ thuật và thay thế chuỗi KBANDS bằng tên các chỉ số thực tế của Bollinger Bands
def replace_bbands(the_list):
    for item in the_list:
        if item == 'KBANDS':
            yield 'Real Lower Band'
            yield 'Real Middle Band'
            yield 'Real Upper Band'
        else:
            yield item
    
# Nhận dữ liệu, mô hình LSTM, danh sách các chỉ số kỹ thuật và khoảng thời gian và trả về giá dự đoán trong tương lai sử dụng mô hình LSTM
def lstm_predict_future(data, model, indicatorArr, period):    
    # data
    data = data[indicatorArr].values
    data = data[-60:]

    # scaled data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(data)

    # model input
    modelInput = scaledData.reshape(-1, scaledData.shape[0], scaledData.shape[1])

    # predicted scaled value
    predictedScaledValue = model.predict(modelInput)

    # predicted value
    predictedValue = scaler.inverse_transform(np.tile(predictedScaledValue, (1, scaledData.shape[1])))[:, 0]
    
    return predictedValue
    
# Nhận dữ liệu, mô hình XGBoost, danh sách các chỉ số kỹ thuật và khoảng thời gian và trả về giá dự đoán trong tương lai sử dụng mô hình XGBoost
def xgboost_predict_future(data, model, indicatorArr, period):
    # indicator
    indicatorArr.insert(1,'volume')
    
    # data
    data = data[indicatorArr]
    data = data[-2:]
    
    # model input
    X = pd.DataFrame({})
    n = len(data)
    for i in range(1, n + 1):
        for column in data.columns:
            X[column + '_date_' + str(i)] = [data.iloc[n - i][column]]
    
    # predicted value
    predictedValue = model.predict(X)
    
    return predictedValue


#Tạo biến df là 1 dataframe chứa dữ liệu từ 1 file csv
dfMSFT = pd.read_csv("../data/MSFT.csv")
dfINTC = pd.read_csv("../data/INTC.csv")

#Định nghĩa giao diện của ứng dụng Dash bằng cách sử dụng các thành phần HTML và Dash components
app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Stock Data', children=[
            html.Div([
                
                html.Div([                
                    html.Button('UPDATE', 
                     id='update_button', 
                     style={"background-color": "#FFFB4F", "border": "none", "color": "black", 
                            "padding": "15px 32px", "text-align": "center", "text-decoration": "none", 
                            "display": "inline-block", "font-size": "16px", 
                            "margin-left": "auto", "margin-top": "10px", 
                            "margin-bottom": "10px", "margin-right": "auto", "width": "20%"})
                ], style={"text-align": "center"}),

                html.Div(id='something', children=''),
                
                html.H1("Stock Price", 
                        style={'textAlign': 'center'}),         
              
                dcc.Dropdown(id='my-dropdown',
                           
                              options=[
                                {'label': 'Intel','value': 'INTC'},
                                {'label': 'Microsoft','value': 'MSFT'},
                                {'label': 'Apple','value': 'AAPL'}, 
                                {'label': 'Tesla','value': 'TSLA'}], 
                             multi=True,
                             value=['MSFT'],
                            #  value=['INTC' , 'MSFT', 'AAPL', 'TSLA'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                
                dcc.Graph(id='stockprice'),
                
                
                html.H1("Stock Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                            #  options=[{'label': 'Microsoft','value': 'MSFT'}], 
                            #  multi=True,
                            #  value=['MSFT'],
                              options=[
                                {'label': 'Intel','value': 'INTC'},
                                {'label': 'Microsoft','value': 'MSFT'},
                                {'label': 'Apple','value': 'AAPL'}, 
                                {'label': 'Tesla','value': 'TSLA'}], 
                             multi=True,
                             value=['MSFT'],
                            #    value=['INTC' , 'MSFT', 'AAPL', 'TSLA'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
                
            ], className="container"),
        ]),
        
        
        dcc.Tab(label='Stock Prediction',children=[
            html.Div([
                
                dcc.Dropdown(id='dropdown-company',
                    options=[
                                {'label': 'Intel','value': 'INTC'},
                                {'label': 'Microsoft','value': 'MSFT'},
                                {'label': 'Apple','value': 'AAPL'}, 
                                {'label': 'Tesla','value': 'TSLA'}], 
                    multi=False,
                    placeholder="Chọn công ty",
                    value='INTC',
                    style={"margin-left": "auto", "margin-top": "10px", "margin-bottom": "10px",
                            "margin-right": "auto", "width": "80%"}),
                
                dcc.Dropdown(id='dropdown-model',
                     options=[{'label': 'Extreme Gradient Boosting (XGBOOST)', 'value': 'XGBOOST'},
                              {'label': 'Recurrent Neural Network (RNN)','value': 'RNN'}, 
                              {'label': 'Long Short Term Memory (LSTM)', 'value': 'LSTM'}], 
                     multi=False, 
                     placeholder="Chọn phương pháp dự đoán",
                     value='LSTM',
                     style={"margin-left": "auto", "margin-top": "10px", "margin-bottom": "10px",
                            "margin-right": "auto", "width": "80%"}),
                
                dcc.Dropdown(id='dropdown-period',
                     options=[{'label': '15 phút', 'value': 15}], 
                     multi=False, placeholder="Chọn khoảng thời gian",
                     value=15,
                     style={"margin-left": "auto", "margin-top": "10px", "margin-bottom": "10px",
                            "margin-right": "auto", "width": "80%"}),
  
                dcc.Dropdown(id='dropdown-indicator',
                     options=[{'label': 'Close Price','value': 'close'},
                              {'label': 'Price Rate of Change (ROC)','value': 'ROC'}, 
                              {'label': 'Relative Strength Index (RSI)', 'value': 'RSI'}, 
                              {'label': 'Simple Moving Averages (SMA)', 'value': 'SMA'},
                              {'label': 'Bolling Bands', 'value': 'KBANDS'}], 
                     multi=True, placeholder="Choose indicators",value=['close'],
                     style={"margin-left": "auto", "margin-top": "10px", "margin-bottom": "10px",
                            "margin-right": "auto", "width": "80%"}),
                
                html.Div([                
                    html.Button('Predict', 
                     id='predict_button', 
                     style={"background-color": "#FFFB4F", "border": "none", "color": "white", 
                            "padding": "15px 32px", "text-align": "center", "text-decoration": "none", 
                            "display": "inline-block", "font-size": "16px", 
                            "margin-left": "auto", "margin-top": "10px", 
                            "margin-bottom": "10px", "margin-right": "auto", "width": "20%"})
                ], style={"text-align": "center"}),

                dcc.Graph(id='predicted_graph')
                
            ])                

        ])


    ])
])



#Nhận vào từ dropdown 'my-dropdown' (Chọn mã cổ phiếu) và trả về biểu đồ giá cổ phiếu dựa trên mã cổ phiếu đã chọn
@app.callback(Output('stockprice', 'figure'),
              [Input('my-dropdown', 'value')]) 
def update_graph(selected_dropdown, df):
    dropdown = {"MSFT": "Microsoft", "INTC": "Intel", "AAPL": "Apple", "TSLA": "Tesla"}
    trace1 = []
    trace2 = []
    trace3 = []
    trace4 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df["date"],
                     y=df["open"],
                     mode='lines', opacity=0.8,
                     name=f'Open {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df["date"],
                     y=df["high"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace3.append(
          go.Scatter(x=df["date"],
                     y=df["low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
        trace4.append(
          go.Scatter(x=df["date"],
                     y=df["close"],
                     mode='lines', opacity=0.5,
                     name=f'Close {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2, trace3, trace4]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout':  go.Layout(colorway=['#0074d9','#ff4136' , '#2ecc40', 
                                             '#ff851b',  '#7fdbff','#b212ca',
                                            '#ffde11', '#001f3f', '#39cccc'] ,
            height=600,
            title=f"Stock Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure

@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value, df):
    dropdown = {"MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df["date"],
                     y=df["volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



#Nhận vào từ các dropdown và nút nhấn trong tab "Stock Prediction" và trả về biểu đồ dự đoán giá cổ phiếu dựa trên mô hình và các chỉ số đã chọn
@app.callback(    
    Output('predicted_graph', 'figure'),
               [Input('predict_button', 'n_clicks')], 
               [
                   State('dropdown-company', 'value'), 
                   State('dropdown-model', 'value'),
                   State('dropdown-indicator', 'value'),
                   State('dropdown-period', 'value')
               ]
              )
def update_graph(n_clicks, companyName, modelName, indicatorArr, period, df):
    data = pd.read_csv("../data/" + companyName + '.csv')
    
    # model
    modelFileName = '../model/' + modelName
    for indicator in indicatorArr:
        if indicator == 'close':
            continue
        if indicator == 'KBANDS':
            indicator = 'BBANDS'
        modelFileName = modelFileName + '_' + indicator
        
    indicatorArr = list(replace_bbands(indicatorArr))

    print(indicatorArr)
    
    predictions = None
    if modelName == 'LSTM' or modelName == 'RNN': 
        modelFileName = modelFileName + '.h5'
        model = load_model(modelFileName)
        futurePredictions = lstm_predict_future(data, model, indicatorArr, period)
        #
        dataset = data
        dataset = dataset[indicatorArr].values
        scaler = MinMaxScaler(feature_range=(0,1))
        dataset = scaler.fit_transform(dataset)
        X = []
        for i in range(60, len(dataset)):
            X.append(dataset[i-60:i][:])
        X = np.array(X[-100:])
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(np.tile(predictions, (1, dataset.shape[1])))[:, 0]
        df = data.iloc[-len(predictions):]
        df['predictions'] = predictions
        #
    elif modelName == 'XGBOOST':
        modelFileName = modelFileName + '.dat'
        model = pickle.load(open(modelFileName, "rb"))
        futurePredictions = xgboost_predict_future(data, model, indicatorArr, period)
        #
        dataset = data
        temp = indicatorArr.copy()
        dataset = dataset[temp]
        for i in range (1, 3):
            for indicator in temp:
                dataset[indicator + "_date_" + str(i)] = dataset[indicator].shift(i)
        dataset.dropna(inplace=True)
        X = dataset.drop(temp, axis=1)
        X = X[-100:]
        predictions = model.predict(X)
        df = data.iloc[-len(predictions):]
        df['predictions'] = predictions
        #
        
    print(modelFileName)
        
    prediction_df = pd.Series(futurePredictions)
    prediction_df = data['close'].append(pd.Series(futurePredictions))
    prediction_df = prediction_df.reset_index()
    prediction_df = prediction_df.drop(columns=['index'], axis=1)
    prediction_df = prediction_df[0]
    prediction_df = prediction_df[-len(futurePredictions):]

    
    figure={
        "data":[
            go.Scatter(
                x=data.index[-300:],
                y=data.close[-300:],
                mode='lines',
                name="Real Price"
            ),
            go.Scatter(
                x=df.index,
                y=df.predictions,
                mode='lines',
                name="Model Validation in 100 previous data points"
            ),
            go.Scatter(
                x=prediction_df.index,
                y=prediction_df.values,
                mode='markers',
                name="Predicted Price"
            ),
        ],
        "layout":go.Layout(
            title=f"Predicted stock price is {prediction_df.values[0]} USD.",
            xaxis={'title':'Data Point'},
            yaxis={'title':'Close Price (USD)'}
        )
    }
    return figure

# #Nhận đầu vào từ nút nhấn update_button và cập nhật dữ liệu cho mã cổ phiếu MSDT (Microsoft) thông qua hàm update_data
@app.callback(Output('something', 'children'), [Input('update_button', 'n_clicks')] )
def update_output(n_clicks):
    update_data('MSFT')
    update_data('INTC')
    dfMSFT = pd.read_csv("../data/MSFT.csv")
    dfINTC = pd.read_csv("../data/INTC.csv")
    

if __name__=='__main__':
    app.run_server( port=8050)