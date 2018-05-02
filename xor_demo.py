from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense (2, activation=’tanh’, input_dim =2))
model.add(Dense (1, activation=’sigmoid’))

model.compile(optimizer=’sgd’, loss=’binary_crossentropy’)

results = model.predict(X)
print(results)
#[[0. 03734707 ][0. 93873811 ][0. 93955719 ][0. 03086801 ]]