import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# Load and preprocess data
dt = pd.read_csv("Crop_Recommendation.csv")
x = dt.iloc[:, :-1].values
y = dt.iloc[:, -1].values
lb = LabelEncoder()
y = lb.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train model
rf = RandomForestClassifier(n_estimators=20, random_state=0)
rf.fit(x_train, y_train)

# Streamlit UI
st.title("Crop Recommendation System")

n = st.number_input("Enter the ratio of Nitrogen in soil:")
p = st.number_input("Enter the ratio of Phosphorus in soil:")
k = st.number_input("Enter the ratio of Potassium in soil:")
t = st.number_input("Enter the Temperature in Celsius:")
h = st.number_input("Enter the Humidity in %:")
ph = st.number_input("Enter the pH Value:")
r = st.number_input("Enter the Rainfall in cm:")

if st.button("Predict"):
    X = np.array([[n, p, k, t, h, ph, r]])
    X = sc.transform(X)
    Y = rf.predict(X)
    crop = lb.inverse_transform(Y)[0]
    st.success(f"The most suitable crop for your conditions is: **{crop}**")
