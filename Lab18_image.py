from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np

# example 8x8 grayscale images
X = np.random.rand(10, 8, 8, 1)
y = np.random.randint(0, 2, size=(10, 1))

inputs = Input(shape=(8,8,1))
x = Conv2D(4, (3,3), activation='relu')(inputs)
x = Flatten()(x)  # this is the embedding
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5)
