import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib

st.set_page_config(page_title="Iris Flower Classifier", layout="centered")

# Load model and preprocessing
model = keras.models.load_model("best_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.title("🌸 Iris Flower Classification App")
st.write("Predict the Iris flower species based on sepal and petal measurements.")
st.markdown("---")

sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.number_input("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_width  = st.number_input("Petal Width (cm)", 0.1, 3.0, 1.3)

if st.button("Predict Flower Type"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)
    label = encoder.inverse_transform([np.argmax(pred)])
    st.success(f"Predicted Species: {label[0]}")
    st.balloons()

st.markdown("---")
st.caption("Developed as Part of Neural Networks with TensorBoard & Bayesian Optimization Project")
