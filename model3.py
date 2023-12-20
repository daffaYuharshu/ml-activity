import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np

def model_aiang3():
# Load the dataset from CSV
    df = pd.read_csv('dataset_new2.csv')

    # Extract features (X) and labels (y)
    X = df[['Work_Hours', 'Sleep_Hours', 'Study_Minutes', 'Break_Time']].values
    y = df['Recommended_Activity'].values

    # Encode categorical labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the model
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Example: Predict recommended activity for input [450, 180, 40, 30, 2]
    input_data = np.array([[450, 180, 40, 70]])
    input_data_scaled = scaler.transform(input_data)

    predicted_activity = model.predict(input_data_scaled)
    predicted_activity_label = label_encoder.classes_[np.argmax(predicted_activity)]
    print("Predicted Recommended Activity:", predicted_activity_label)
    print("Predicted Recommended Activity:", predicted_activity)
    print("Predicted Recommended Activity:", y_encoded)
    return model
if __name__ == '__main__':
    model = model_aiang3()
    model.save("model_aiang3.h5")
