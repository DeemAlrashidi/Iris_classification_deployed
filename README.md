# 🌸 Iris Flower Classification — ML + Streamlit App

An end-to-end machine learning project built using the classic **Iris dataset**, including **data preprocessing**, **Bayesian optimization**, and a **Streamlit web app** for interactive species prediction.

---

## 📘 Overview
This project classifies iris flowers into one of three species:
- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

It follows a complete ML workflow:
- Data preprocessing and scaling  
- Neural network training with **TensorFlow + Keras**  
- **Bayesian hyperparameter tuning**  
- **TensorBoard visualization**  
- **Streamlit deployment** for live predictions  

---

## ⚙️ Features
- **Model Training:** Neural Network trained on Iris dataset  
- **Bayesian Optimization:** Automated hyperparameter tuning  
- **TensorBoard Integration:** Real-time metric visualization  
- **Streamlit App:** Predicts flower species interactively  
- **Pretrained Artifacts:** Includes model, scaler, and encoder  

---

## 📁 Repository Structure
├── IRIS.csv # Dataset
├── IRIS_final.ipynb # Notebook (EDA + model + tuning)
├── app.py # Streamlit app for prediction
├── best_model.keras # Saved tuned model
├── scaler.pkl # StandardScaler used during training
├── encoder.pkl # LabelEncoder for species names
└── README.md # Project documentation


---

## ▶️ Quickstart

### Run Locally
```bash
# Clone repository
git clone https://github.com/yourusername/iris-ml-app.git
cd iris-ml-app

# Create and activate environment (optional)
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install streamlit tensorflow scikit-learn joblib numpy pandas

# Run the app
streamlit run app.py
```
---
## 🖥️ Streamlit App Interface

Input sliders for:

 - Sepal Length (cm)

 - Sepal Width (cm)

 - Petal Length (cm)

 - Petal Width (cm)

* Predicts flower type instantly

* Displays prediction and success animation


## 🎥 Demo (View raw will download the video locally)
[▶ Watch the demo](Demo_iris.mp4)
