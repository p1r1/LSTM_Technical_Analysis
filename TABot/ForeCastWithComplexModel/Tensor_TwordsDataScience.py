import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from ForeCastWithComplexModel.Tensor_TowardsDataScience_Functions import train_test_split, line_plot, prepare_data, build_lstm_model

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

train, test = train_test_split(hist, test_size=0.2)

line_plot(train[target_col], test[target_col], 'training', 'test', title='')

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

model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)

history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)

preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
line_plot(targets, preds, 'actual', 'prediction', lw=3)

model.save(f"""ethusdt_{datetime.now().strftime('%Y-%m-%d')}.h5""")

#############################################
# for i in range(len(preds)):
#     print(f"""{targets[i]} xXx {preds[i]}""")
print(preds)
