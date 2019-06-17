import pandas as pd
import numpy as np
import tensorflow as tf

from utils import get_data


def crash_proof():
    """
    in case of GPU CUDA crashing
    """
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)


price_type = 'close'
symbol = '600000.SH'
df = get_data(symbol_=symbol)

test_len = 120
timesteps = 60
training_set = df[:-test_len][price_type]
training_set = pd.DataFrame(training_set)

# scaled
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# to_supervise
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(timesteps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - timesteps:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print('here')

# building network

from keras.models import load_model

try:
    regressor = load_model('ashare_lstm_test.h5')

except:
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units=1))

    regressor.summary()

    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs=50, batch_size=32)

print('here')

# test

test_set = df[-test_len:][price_type]
test_set = pd.DataFrame(test_set)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
test_set_scaled = sc.fit_transform(test_set)
test_set_scaled = pd.DataFrame(test_set_scaled)

# Getting the predicted stock price of 2017
dataset_total = df[price_type]
inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(timesteps, timesteps + test_len):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
import matplotlib.pyplot as plt

plt.plot(test_set.values, color='red', label=f'Real Price-{symbol}')
plt.plot(predicted_stock_price, color='blue', label=f'Predicted Price-{symbol}')
plt.title(f'{symbol}-Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

result = pd.DataFrame()
result['real'] = np.squeeze(test_set)
result['predict'] = np.squeeze(predicted_stock_price)
result['date'] = list(df[-test_len:]['date'].values)

result.to_csv('tt.csv')

# 预测结果类似MA
