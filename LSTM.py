import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
#DATA SETUP
#1. Split the data




#MODEL SETUP
#1. Initialize Model
model = Sequential()

#2. Add first LSTM layer [Input Layer] Todo: set correct input shape according to the data frame input
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(n, m)))

#3. Deepen the model with additional layers
model.add(LSTM(units=50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, activation='relu'))
model.add(Dropout(0.2))

#4. Add Dense layer to interpret features 
model.add(Dense(units=25))

#5. Output Layer
model.add(Dense(units=1))

#6. Compile the model and set loss function ToDo: figure out best loss function to set
model.compile(optimizer='adam', loss='mean_squared_error')


