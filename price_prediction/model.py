from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
class LSTM_4:
    def __init__(self, trainX, trainY):
        self.model = Sequential()
        self.trainX = trainX
        self.trainY = trainY

    def create_model(self):
        self.model.add(LSTM(4, input_shape=(1, 1)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self):
        self.model.fit(self.trainX, self.trainY, epochs=100, batch_size=1, verbose=2)

    def predict(self, X):
        output = self.model.predict(X)
        return output
i = [1,2,3,4,5,6]
df = pd.DataFrame({'one':i})
df = df['one'].to_numpy()
print(df.reshape(6,1,1))