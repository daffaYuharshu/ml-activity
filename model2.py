import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def model_aiang2():
# Load the dataset from CSV
    df = pd.read_csv('dataset_new2.csv')

    # Extract features (X) and labels (y)
    X = df[['Work_Hours', 'Sleep_Hours', 'Study_Minutes', 'Break_Time']].values
    y = df['Stress_Level'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Example: Predict stress level for input [450, 180, 40, 30]
    input_data = np.array([[450, 50, 10, 30]])
    input_data_scaled = scaler.transform(input_data)

    predicted_stress_level = model.predict(input_data_scaled)
    rounded_stress_level = int(round(predicted_stress_level[0, 0]))

    # Map the continuous range to the desired integer range (1 to 5)
    return model
if __name__ == '__main__':
    model = model_aiang2()
    model.save("model_aiang2.h5")