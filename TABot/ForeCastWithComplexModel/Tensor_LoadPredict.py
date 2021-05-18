import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from ForeCastWithComplexModel.Tensor_TowardsDataScience_Functions import train_test_split, line_plot, prepare_data

hist = pd.read_json('Binance_ETHUSDT_1d_2years.json', orient='records')
hist.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore']

del hist['open']
del hist['high']
del hist['low']
del hist['open_time']
# del hist['volume']
# del hist['quote_asset_volume']
del hist['number_of_trades']
del hist['taker_buy_base_asset_volume']
del hist['taker_buy_quote_asset_volume']
del hist['ignore']

hist.set_index('close_time')  # drop=True, inplace=True
hist.index = pd.to_datetime(hist['close_time'], unit='ms')

del hist['close_time']
target_col = 'close'

# print(hist.head(5))
# print(type(hist.index))
for i in range(5):
    last_date = hist.iloc[[-1]].index
    last_close = hist.iloc[-1]['close']
    last_date = last_date + pd.Timedelta(days=1)
    #last_date = pd.to_datetime(last_date, format="%Y-%m-%d %H:%M:%S.%f")
    hist = hist.append(pd.DataFrame({'close': last_close, 'volume': 1, 'quote_asset_volume': 1}, index=last_date))
    # hist = hist.append(pd.DataFrame(index=last_date))

    #hist.loc[last_date] = new_row

print(hist.tail(5))

# last_date = hist.iloc[[-1]].index
# last_date = last_date + pd.Timedelta(days=1)
# print(pd.DataFrame({'close': 200}, index=last_date))

# row = pd.Series({'close_time': '2020-11-07 23:59:59.999'})
# row2 = pd.Series({'close_time': '2020-11-08 23:59:59.999'})
# row3 = pd.Series({'close_time': '2020-11-09 23:59:59.999'})
# hist.append(row)
# hist.append(row2)
# hist.append(row3)
# hist.append(pd.DataFrame(index=[last_date]))

train, test = train_test_split(hist, test_size=0.2)

# line_plot(train[target_col], test[target_col], 'training', 'test', title='')

np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 100
epochs = 500
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'

train, test, X_train, X_test, y_train, y_test = prepare_data(
    hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

model = load_model('ethusdt_2020-11-06.h5', compile=True)

targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)

preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
line_plot(targets, preds, 'actual', 'prediction', lw=3)

####################################
print(preds.tail(10))

