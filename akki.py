import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

st.title('Diabetes Prediction systeam')
st.write('Develop a web application that predicts whether a person has diabetes based on input parameters such as pregnancies, glucose levels, blood pressure, etc.')

# Load dataset
df = pd.read_csv("diabetes.csv")

# Sidebar layout
st.sidebar.title("Enter Patient Details")

# Select input method: sliders or dropdowns
input_method = st.sidebar.radio("Select Input Method:", ("Sliders", "Dropdowns"))

# Input fields based on selected method
if input_method == "Sliders":
    # Use sliders for input
    Pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 0, 1)
    Glucose = st.sidebar.slider("Glucose", 0, 300, 100, 1)
    BloodPressure = st.sidebar.slider("Blood Pressure", 0, 200, 72, 1)
    SkinThickness = st.sidebar.slider("Skin Thickness", 0, 100, 20, 1)
    Insulin = st.sidebar.slider("Insulin", 0, 500, 79, 1)
    BMI = st.sidebar.slider("BMI", 0.0, 60.0, 32.0, 0.1)
    DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.0, 0.5, 0.1)
    Age = st.sidebar.slider("Age", 0, 120, 30, 1)
else:
    # Use sliders for all inputs as selectbox can't handle floats
    Pregnancies = st.sidebar.selectbox("Pregnancies", range(21), 0)
    Glucose = st.sidebar.selectbox("Glucose", range(301), 100)
    BloodPressure = st.sidebar.selectbox("Blood Pressure", range(201), 72)
    SkinThickness = st.sidebar.selectbox("Skin Thickness", range(101), 20)
    Insulin = st.sidebar.selectbox("Insulin", range(501), 79)
    BMI = st.sidebar.slider("BMI", 0.0, 60.0, 32.0, 0.1)
    DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.0, 0.5, 0.1)
    Age = st.sidebar.selectbox("Age", range(121), 30)

# Create a list for the input data
input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]

# Example: Using KNeighborsClassifier
rf = KNeighborsClassifier(n_neighbors=18)

# Train the model
x = df[
    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
rf.fit(xtrain, ytrain)


# Predict function
def predict_diabetes(input_data):
    prediction = rf.predict(input_data)
    return prediction


# Adding a predict button
if st.sidebar.button("Predict"):
    prediction = predict_diabetes(input_data)

    # Output the result based on the prediction
    if prediction == 0:
        st.error('This person does not have diabetes.')
    else:
        st.success('This person has diabetes.')

    # Calculate accuracy (just for demonstration)
    y_pred = rf.predict(xtest)
    accuracy = round(accuracy_score(ytest, y_pred) * 100, 2)
    st.write(f'Accuracy on test data: {accuracy}%')
