from flask import Flask, jsonify,request
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import load_model

app = Flask(__name__)
model =load_model("model_aiang2.h5")
model2 =load_model("model_aiang3.h5")
df = pd.read_csv('dataset_new2.csv')
scaler = StandardScaler()
label_encoder = LabelEncoder()
X = df[["Work_Hours", "Sleep_Hours", "Study_Minutes", "Break_Time"]]
y = df["Stress_Level"]
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
z = df['Recommended_Activity'].values
y_encoded = label_encoder.fit_transform(z)
# Fit the scaler on the training data
scaler.fit(X_train)
@app.route("/")
def index():
    return jsonify({
        "status":{
            "code":200,
            "message":"Success fetching the API",
        },
        "data":None
    }),200
@app.route("/prediction",methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        data = request.get_json(force=True) 
        input_data=[[data[key] for key in ["Work_Hours", "Sleep_Hours", "Study_Minutes", "Break_Time"]]]
        y = df["Stress_Level"]
        y_encoded = label_encoder.fit_transform(y)
        stresslevel = model.predict(scaler.transform(input_data))
        rounded_stress_level = int(round(stresslevel[0, 0]))
        z = df['Recommended_Activity'].values
        z_encoded = label_encoder.fit_transform(z)
        # input_data.append(rounded_stress_level)
        predicted_activity = model2.predict(input_data)
        predicted_activity_label = label_encoder.classes_[np.argmax(predicted_activity)]
        # print(predicted_activity_label)
        # Load the trained model and scale
        return jsonify({'Stress Level':rounded_stress_level,'Rekomendasi Aktivitas':predicted_activity_label})
    else:
        return jsonify({
            "status":{
                "code":405,
                "message":"invalid method"
            },
            "data":None,
        }), 405