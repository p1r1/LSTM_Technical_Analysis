import json
import os
from binance.client import Client
from datetime import date, datetime


def add_years(d, years):
    """Return a date that's `years` years after the date (or datetime)
    object `d`. Return the same calendar date (month and day) in the
    destination year, if it exists, otherwise use the following day
    (thus changing February 29 to March 1).
    XRP
    """
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))


def SaveBinanceDataToFile(kline_data):
    """
    :param kline_data:
    :type: Client.get_historical_klines
    :return: file_path
    """
    # open a file with filename including symbol, interval and start and end converted to milliseconds
    file_name = "Binance_{}_{}_{}.json".format(
        symbol,
        interval,
        (datetime.now()).strftime("%d-%m-%Y_%H-%M-%S")
    )
    with open(file_name, 'w') as f:  # w -> set file write mode
        f.write(json.dumps(kline_data))

    return os.path.abspath(file_name)


symbol = "BTCUSDT"
# start = add_years(datetime.now(), 2).strftime("%Y-%m-%d %H:%M:%S") # "1 Nov 2020"
# end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
interval = Client.KLINE_INTERVAL_1DAY
client = Client()

# klines = get_historical_klines(symbol, interval, start, end)
# klines = Client.get_historical_klines(client, symbol, interval, start, end, 1000)
klines = Client.get_historical_klines(self=client, symbol=symbol, interval=interval, start_str="2 year ago UTC",
                                      limit=1000)
print(SaveBinanceDataToFile(klines))

# SAVE
# open a file with filename including symbol, interval and start and end converted to milliseconds
# fileName = "Binance_{}_{}_{}.json".format(
#     symbol,
#     interval,
#     # date_to_milliseconds(start),
#     # date_to_milliseconds(end)
#     "2years")
# with open(fileName, 'w') as f:  # w -> set file write mode
#     f.write(json.dumps(klines))

# import json
# from binance.client import Client
#
# client = Client("", "")
#
# klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

# print(fileName)
