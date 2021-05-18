from datetime import datetime
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go  # better
from dash import dash
import dash_html_components as html
import dash_core_components as dcc


def DoTheThing(model_name, file_path, target_column='close', target_date='close_time',
               look_back_size=7, batch_size=30, epochs_count=720):
    """
    :param model_name saved model name
    :param file_path: binance data file path
    :param target_column: open, high, low, close
    :param target_date: open_time or close_time
    :param look_back_size: for shifting data
    :param batch_size: örneklem
    :param epochs_count: iterations
    :return: nothing - draws graph
    """

    #################
    """read file"""
    #################
    # df = pd.read_json('../Binance_BTCUSDT_1d_2years.json', orient='records')
    df = pd.read_json(file_path)
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                  'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                  'taker_buy_quote_asset_volume', 'ignore']

    # inlace = True means play with original data don't makecopy of it
    df[target_date] = pd.to_datetime(df[target_date], unit='ms')
    df.set_axis(df[target_date], inplace=True)
    df.drop(columns=['open', 'high', 'low', 'open_time', 'volume', 'quote_asset_volume', 'number_of_trades',
                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], inplace=True)

    df = df.iloc[:-1]  # remove today bcs today is not closed YET!

    # print(df.tail(5))

    #################
    """shift data"""
    #################
    close_data = df[target_column].values
    close_data = close_data.reshape((-1, 1))

    # [2, 3, 4, 5, 4, 6, 7, 6, 8, 9]
    # the required data format (n=3) would be this:
    # [2, 3, 4] -> [5]
    # [3, 4, 5] -> [4]
    # [4, 5, 4] -> [6]
    # [5, 4, 6] -> [7]
    # [4, 6, 7] -> [6]
    # [6, 7, 6] -> [8]
    # [7, 6, 8] -> [9]

    ##################################
    """split data to test and train"""
    ##################################
    split_percent = 0.80
    split = int(split_percent * len(close_data))

    close_train = close_data[:split]
    close_test = close_data[split:]

    date_train = df[target_date][:split]
    date_test = df[target_date][split:]

    print(len(close_train))
    print(len(close_test))

    ########################
    """create time series"""
    ########################
    look_back = look_back_size

    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=batch_size)
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

    ################
    """LSTM model"""
    #################
    # weak model tweak this maybe merge with others ?
    model = Sequential()
    model.add(
        LSTM(100,
             activation='relu',
             input_shape=(look_back, 1))
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = epochs_count
    ##################
    """the training"""
    ##################
    # model.fit_generator(train_generator, epochs=num_epochs, verbose=1) # deprecated
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

    ###################
    """predict current"""
    ###################
    # prediction = model.predict_generator(test_generator) # deprecated
    prediction = model.predict(test_generator)

    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))

    # plot
    trace1 = go.Scatter(
        x=date_train,
        y=close_train,
        mode='lines',
        name='Data'
    )
    trace2 = go.Scatter(
        x=date_test,
        y=prediction,
        mode='lines',
        name='Prediction'
    )
    trace3 = go.Scatter(
        x=date_test,
        y=close_test,
        mode='lines',
        name='Ground Truth'
    )
    layout = go.Layout(
        title="ETHUSDT",
        xaxis={'title': "Date"},
        yaxis={'title': "Close"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.show()

    ##################################
    """predict future - forecasting"""
    ##################################
    close_data = close_data.reshape((-1))

    def predict(num_prediction, model):
        prediction_list = close_data[-look_back:]

        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back - 1:]

        return prediction_list

    def predict_dates(num_prediction):
        last_date = df[target_date].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
        return prediction_dates

    num_prediction = 30
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    ########################
    """save trained model"""
    ########################

    model.save(f"""{model_name}.h5""")

    # print(forecast)
    # print(forecast_dates)

    ######################
    """plot function"""
    ######################
    def plotGraph(title, close_predict, date_predict):
        # #  plot
        # if close_actual is None:
        #     close_actual =
        trace1_x = go.Scatter(
            x=df[target_date],
            y=close_data,
            mode='lines',
            name='Data'
        )
        trace2_x = go.Scatter(
            x=date_predict,
            y=close_predict,
            mode='lines',
            name='Prediction'
        )
        # trace3_x = go.Scatter(
        #     x=date_test,
        #     y=close_test,
        #     mode='lines',
        #     name='Ground Truth'
        # )
        layout_x = go.Layout(
            title=title,
            xaxis={'title': "Date"},
            yaxis={'title': "Close"}
        )
        # fig_x = go.Figure(data=[trace1_x, trace2_x], layout=layout_x)
        # fig_x.show()

        # app = dash.Dash(__name__)
        app = dash.Dash()
        # server = app.server
        app.layout = html.Div(children=[
            # html.H1(children='Sales Funnel Report'),
            # html.Div(children='''National Sales Funnel Report.'''),
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [trace1_x, trace2_x],
                    'layout':
                    # go.Layout(title='Order Status by Customer', barmode='stack')
                        layout_x
                })
        ])
        app.run_server(debug=False, port=8066, host='0.0.0.0')
        # if __name__ == '__main__':
        #     app.run_server(debug=False, port=8066, host='0.0.0.0')

    ####################
    """plot the graph"""
    ####################

    assert len(forecast_dates) == len(forecast)

    for i in range(len(forecast_dates)):
        print(f"""{forecast_dates[i]: {forecast[i]}}""")

    print("http://192.168.1.117:8066")
    plotGraph('ETHUSDT', forecast, forecast_dates)

    #TODO:
    # If there exists any problem during the training of the model, try tuning these parameters:
    # look_back
    # batch size
    # LSTM units
    # num_epochs
    # You can also enhance the architecture by stacking LSTM layers.
