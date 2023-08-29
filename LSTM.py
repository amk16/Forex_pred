import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
#DATA SETUP
#1. Get Forex data
    #Todo: import the data and sort as needed

#2. Split the data
X_train, X_temp, y_train, y_temp = train_test_split(data, targets, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



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

#TRAIN MODEL
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

#PLOTTING
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.show()

#TEST MODEL
y_pred = model.predict(X_test)

# CALCULATE METRICS
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)



