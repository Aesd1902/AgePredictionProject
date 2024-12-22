import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Define a simple CNN model for age prediction
def create_age_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for age prediction (regression)
    return model

# Compile and return the model
model = create_age_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Load and preprocess your dataset (placeholder function for now)
def load_your_data():
    # Placeholder for actual dataset loading
    # Replace this function with the actual data loading and preprocessing
    X = np.random.rand(1000, 64, 64, 3)  # Random data simulating face images
    y = np.random.randint(0, 100, size=(1000,))  # Random ages between 0-100
    return X, y

# Load the dataset
X, y = load_your_data()

# Train-validation split (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with the dataset
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Create the directory to save the model if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the trained model
model.save('model/age_model.h5')

print("Model training completed and saved successfully.")
