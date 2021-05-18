# date,open,close,high,low,adjusted_close,volume
# 2017-01-03,34.98,35.15,35.57,34.84,32.4292294525,10904891
# 2017-01-04,35.6,37.09,37.235,35.47,34.2190645915,23381982
# 2017-01-05,37.01,36.39,37.05,36.065,33.5732477888,15635155

# Load CSV data into a dataframe
# Rescale
from sklearn import preprocessing
import pandas
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot
from numpy import concatenate
from sklearn.preprocessing import StandardScaler

# json = json.load('Binance_BCHUSDT_1M_1543622400000-1604188800000.json')
# dataframe = pandas.DataFrame(json, orient='split')

sample_quantitiy = 1000
dataframe = pandas.read_json('Binance_ETHUSDT_1d_2years.json', orient='records')
dataframe.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                     'taker_buy_quote_asset_volume', 'ignore']
dataframe.set_index('close_time') # drop=True, inplace=True
# sample_quantitiy = len(dataframe) - 2

# Add to predict column (adjusted close) and shift it. This is our output
dataframe['output'] = dataframe.target_column.shift(-1)  # - change ofm to close
# Remove NaN on the final sample (because we don't have tomorrow's output)
dataframe = dataframe.dropna()

# scaler = preprocessing.MinMaxScaler()  #default is 0,1 -change OFM
# scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # -change OFM
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # -change OFM
# scaler = StandardScaler() # standart dağılım -for bell curve (guassian)
rescaled = scaler.fit_transform(dataframe.values)

#############################
# Split into training/testing
training_ratio = 0.8
training_testing_index = int(len(rescaled) * training_ratio)
training_data = rescaled[:training_testing_index]
testing_data = rescaled[training_testing_index:]
training_length = len(training_data)
testing_length = len(testing_data)

# Split training into input/output. Output is the one we added to the end
training_input_data = training_data[:, 0:-1]
training_output_data = training_data[:, -1]

# Split testing into input/output. Output is the one we added to the end
testing_input_data = testing_data[:, 0:-1]
testing_output_data = testing_data[:, -1]

###############################################
# Reshape data for (Sample, Timesteps, Features)
training_input_data = training_input_data.reshape(training_input_data.shape[0], 1, training_input_data.shape[1])
testing_input_data = testing_input_data.reshape(testing_input_data.shape[0], 1, testing_input_data.shape[1])

# Build the model
model = Sequential()
model.add(LSTM(100, input_shape=(training_input_data.shape[1], training_input_data.shape[2])))  # stateful=True,
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#################################################
# Fit model with history to check for overfitting
history = model.fit(
    training_input_data,
    training_output_data,
    epochs=sample_quantitiy,
    validation_data=(testing_input_data, testing_output_data),
    shuffle=False
)

pyplot.plot(history.history['loss'], label='Training Loss')
pyplot.plot(history.history['val_loss'], label='Testing Loss')
pyplot.legend()
pyplot.show()

######################
# Generate predictions
print("Generate predictions")
raw_predictions = model.predict(testing_input_data)

#######################################
# Reshape testing input data back to 2d
testing_input_data = testing_input_data.reshape((testing_input_data.shape[0], testing_input_data.shape[2]))
testing_output_data = testing_output_data.reshape((len(testing_output_data), 1))

####################################
# Invert scaling for prediction data
unscaled_predictions = concatenate((testing_input_data, raw_predictions), axis=1)
unscaled_predictions = scaler.inverse_transform(unscaled_predictions)
unscaled_predictions = unscaled_predictions[:, -1]

# Invert scaling for actual data
unscaled_actual_data = concatenate((testing_input_data, testing_output_data), axis=1)
unscaled_actual_data = scaler.inverse_transform(unscaled_actual_data)
unscaled_actual_data = unscaled_actual_data[:, -1]

# Plot prediction vs actual
pyplot.plot(unscaled_actual_data, label='Actual Close')
pyplot.plot(unscaled_predictions, label='Predicted Close')
pyplot.legend()
pyplot.show()

future = []
currentStep = raw_predictions[-1]  # last step from the previous prediction
future_pred_count = 10  # 10 candlesticks

for i in range(future_pred_count):
    currentStep = model.predict(currentStep)  # get the next step
    future.append(currentStep)  # store the future steps

# after processing a sequence, reset the states for safety
model.reset_states()

print("yow")
c = 0
for i in range(len(unscaled_actual_data)):
    print(f"""{str(c)}. actual close:{unscaled_actual_data[i]} xXx predicted close:{unscaled_predictions[i]}""")
    c = c + 1

print("future 10:")
print(future)
#
# # date,open,close,high,low,adjusted_close,volume
# # 2017-01-03,34.98,35.15,35.57,34.84,32.4292294525,10904891
# # 2017-01-04,35.6,37.09,37.235,35.47,34.2190645915,23381982
# # 2017-01-05,37.01,36.39,37.05,36.065,33.5732477888,15635155
#
# # Load CSV data into a dataframe
# # Rescale
# from sklearn import preprocessing
# import pandas
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# from matplotlib import pyplot
# from numpy import concatenate
# from sklearn.preprocessing import StandardScaler
#
# # json = json.load('Binance_BCHUSDT_1M_1543622400000-1604188800000.json')
# # dataframe = pandas.DataFrame(json, orient='split')
#
# sample_quantitiy = 1000
# dataframe = pandas.read_json('Binance_ETHUSDT_1d_2years.json', orient='records')
# dataframe.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
#                      'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
#                      'taker_buy_quote_asset_volume', 'ignore']
# dataframe.set_index('close_time')
# # sample_quantitiy = len(dataframe) - 2
#
# # Add to predict column (adjusted close) and shift it. This is our output
# dataframe['output'] = dataframe.close.shift(-1)  # - change ofm to close
# # Remove NaN on the final sample (because we don't have tomorrow's output)
# dataframe = dataframe.dropna()
#
# scaler = preprocessing.MinMaxScaler()  # -change OFM
# # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # -change OFM
# # scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))  # -change OFM
# # scaler = StandardScaler() # standart dağılım -for bell curve (guassian)
# rescaled = scaler.fit_transform(dataframe.values)
#
# #############################
# # Split into training/testing
# training_ratio = 0.8
# training_testing_index = int(len(rescaled) * training_ratio)
# training_data = rescaled[:training_testing_index]
# testing_data = rescaled[training_testing_index:]
# training_length = len(training_data)
# testing_length = len(testing_data)
#
# # Split training into input/output. Output is the one we added to the end
# training_input_data = training_data[:, 0:-1]
# training_output_data = training_data[:, -1]
#
# # Split testing into input/output. Output is the one we added to the end
# testing_input_data = testing_data[:, 0:-1]
# testing_output_data = testing_data[:, -1]
#
# ###############################################
# # Reshape data for (Sample, Timesteps, Features)
# training_input_data = training_input_data.reshape(training_input_data.shape[0], 1, training_input_data.shape[1])
# testing_input_data = testing_input_data.reshape(testing_input_data.shape[0], 1, testing_input_data.shape[1])
#
# # Build the model
# model = Sequential()
# model.add(LSTM(100, input_shape=(training_input_data.shape[1], training_input_data.shape[2])))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
#
# #################################################
# # Fit model with history to check for overfitting
# history = model.fit(
#     training_input_data,
#     training_output_data,
#     epochs=sample_quantitiy,
#     validation_data=(testing_input_data, testing_output_data),
#     shuffle=False
# )
#
# pyplot.plot(history.history['loss'], label='Training Loss')
# pyplot.plot(history.history['val_loss'], label='Testing Loss')
# pyplot.legend()
# pyplot.show()
#
# ######################
# # Generate predictions
# print("Generate predictions")
# raw_predictions = model.predict(testing_input_data)
#
# #######################################
# # Reshape testing input data back to 2d
# testing_input_data = testing_input_data.reshape((testing_input_data.shape[0], testing_input_data.shape[2]))
# testing_output_data = testing_output_data.reshape((len(testing_output_data), 1))
#
# ####################################
# # Invert scaling for prediction data
# unscaled_predictions = concatenate((testing_input_data, raw_predictions), axis=1)
# unscaled_predictions = scaler.inverse_transform(unscaled_predictions)
# unscaled_predictions = unscaled_predictions[:, -1]
#
# # Invert scaling for actual data
# unscaled_actual_data = concatenate((testing_input_data, testing_output_data), axis=1)
# unscaled_actual_data = scaler.inverse_transform(unscaled_actual_data)
# unscaled_actual_data = unscaled_actual_data[:, -1]
#
# # Plot prediction vs actual
# pyplot.plot(unscaled_actual_data, label='Actual Close')
# pyplot.plot(unscaled_predictions, label='Predicted Close')
# pyplot.legend()
# pyplot.show()
#
# print("yow")
# c = 0
# for i in range(len(unscaled_actual_data)):
#     print(f"""{str(c)}. actual close:{unscaled_actual_data[i]} xXx predicted close:{unscaled_predictions[i]}""")
#     c = c + 1
#
