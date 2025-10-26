from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import numpy as np

# example protein sequences (1=A, 2=C, 3=D, ...)
X = np.array([[1,2,3], [2,3,1]])
y = np.array([0, 1])

model = Sequential([
    Embedding(input_dim=4, output_dim=3, input_length=3),
    Flatten(),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5)
