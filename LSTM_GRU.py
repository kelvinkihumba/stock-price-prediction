import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

'''
apple = yf.Ticker("AAPL")
apple = apple.history(period="8y")
apple.to_csv("apple.csv")
apple.plot.line(y="Close", use_index=True)
plt.show()'''
data = pd.read_csv("apple.csv")
data.plot.line(y="Close", use_index=True)
plt.show()
dataframe = data[['Close']]
df = dataframe.to_numpy()

scaler = MinMaxScaler()
df = scaler.fit_transform(df)

from tensorflow import keras
import numpy as np

num_train = int(0.65*len(df))
num_val = int(0.85*len(df))

train_data = keras.utils.timeseries_dataset_from_array(
    df[:-2],
    targets=df[2:],
    sequence_length=2,
    batch_size=500,
    start_index=0,
    end_index=num_train)

val_data = keras.utils.timeseries_dataset_from_array(
    df[:-2],
    targets=df[2:],
    sequence_length=2,
    shuffle=True,
    batch_size=500,
    start_index=num_train,
    end_index=num_val)

test_data = keras.utils.timeseries_dataset_from_array(
    df[:-2],
    targets=df[2:],
    sequence_length=2,
    shuffle=True,
    batch_size=500,
    start_index=num_val)

for samples, targets in train_data:
    print("samples shape:", samples.shape)
    print("targets shape:", targets.shape)
    break
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

model = Sequential([layers.Input((2, 1)),
                    #layers.Conv1D(16, 3, activation="relu"),
                    #layers.Conv1D(32, kernel_size=2, strides=1),
                    #layers.GRU(128, recurrent_dropout=0.02, return_sequences=True),
                    layers.GRU(64, recurrent_dropout=0.25),
                    #layers.GRU(32, recurrent_dropout=0.25),
                    #layers.LSTM(64, recurrent_dropout=0.25),
                    #layers.Bidirectional(layers.LSTM(64, recurrent_dropout=0.25)),
                    #layers.Flatten(),
                    #layers.Dropout(0.5),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mae'])

model.fit(train_data, validation_data=val_data, epochs=1000) 

#print(model.summary())

train_predictions = model.predict(df).flatten()
date = np.arange(len(df))
y_train = df

plt.plot(date, train_predictions)
plt.plot(date, y_train)
plt.legend(['Prediction', 'Observation'])
plt.show()
