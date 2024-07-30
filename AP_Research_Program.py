#section 1
import pandas as pd
from tensorflow import keras
from keras import layers
import numpy as np

#section #2
df = pd.read_csv('AP_Research/data.csv')
df = df.sample(frac=1).reset_index(drop=True)


#section 3
input_factors = ['California Statewide GHG Emissions(MMTCO2 Eq.)', 
                 'Los Angeles County GHG Emissions(MMTCO2 Eq.)', 
                 'Population']
training_input = df.loc[8:24, input_factors]
training_output = df.loc[8:24, '>100 AQI Day Count']
testing_input = df.loc[25:32, input_factors]
testing_output = df.loc[25:32, '>100 AQI Day Count']
final_testing_input = df.loc[0:7, input_factors]
final_testing_output = df.loc[0:7, '>100 AQI Day Count']

#section 4
model = keras.Sequential([
    layers.BatchNormalization(input_shape=[3]),
    layers.Dense(6, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.3),  # Dropout layer to prevent overfitting
    layers.Dense(1),])

#section 5
model.compile(optimizer='adam', loss='mae')

#section 6
history = model.fit(
    training_input, training_output,
    validation_data=(testing_input, testing_output),
    batch_size=3,
    epochs=25,
    verbose=0)

#section 7
accuracy = 0

for i in range(8):
    input_data = np.array([[final_testing_input.iloc[i,0] , 
                            final_testing_input.iloc[i,1], 
                            final_testing_input.iloc[i,2]]])
    p = model.predict(input_data)
    prediction = p[0][0]
    actual = final_testing_output[i]
    difference = abs(actual-prediction)
    accuracy += 1-difference/actual

print(str(round((100*(accuracy/8)),2))+'% Accuracy')