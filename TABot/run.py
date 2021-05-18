import os
from binance.client import Client
from GetDataFromBinance.historical_data import SaveBinanceDataToFile
from ForeCastWithTimeSeries import Time_Series_Mine

##############################################################################################
# start = add_years(datetime.now(), 2).strftime("%Y-%m-%d %H:%M:%S") # "1 Nov 2020"
# end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# interval = Client.KLINE_INTERVAL_1DAY
# interval = Client.KLINE_INTERVAL_15MINUTE
# client = Client()

# klines = get_historical_klines(symbol, interval, start, end)

# klines = Client.get_historical_klines(self=Client(), symbol=symbol, interval=interval,
#                                       start_str="3 year ago UTC", end_str="1 Nov 2020", limit=1000)

# klines = Client.get_historical_klines(self=Client(), symbol=binance_symbol, interval=interval,
#                                       start_str="3 year ago UTC", limit=1000)

# print(SaveBinanceDataToFile(klines))
################################################################################################

symbol = "XRPUSDT"
interval = "15m"
target_column = 'close'
target_date = 'close_time'
look_back_size = 4
batch_size = 24
epochs_count = 100
file_name = f"""{symbol}_{interval}_{target_column}_{target_date}_{look_back_size}_{batch_size}_{epochs_count}"""

klines = Client.get_historical_klines(self=Client(), symbol=symbol, interval=interval,
                                      start_str="01.01.2020", end_str="19.11.2020", limit=1000)

binance_data_file = SaveBinanceDataToFile(klines)
Time_Series_Mine.DoTheThing(model_name=file_name, file_path=binance_data_file, target_column='close',
                            target_date='close_time', look_back_size=4, batch_size=24, epochs_count=100)

if os.path.exists(binance_data_file):
    os.remove(binance_data_file)
