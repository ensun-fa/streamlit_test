# Import libraries
import streamlit as st
import numpy as np
import pickle

# Create lists with range of options for the drop down boxes
year_choices = [x for x in range(60, 86)]
cyl_choices = [x for x in range(2, 9)]

# Load the model
with open("./files/model.pkl", "rb") as f:
    model = pickle.load(f)

with st.sidebar:
    st.write("Input parameters to the model:")
    mpg = st.text_input("MPG", value="0")
    disp = st.text_input("Displacement", value="0")
    wgt = st.text_input("Weight", value="0")
    accel = st.text_input("Acceleration", value="0")
    year = st.selectbox("Year", year_choices)
    cyl = st.selectbox("Cylinder", cyl_choices)

# Predict with the model
new_data = np.array([mpg, disp, wgt, accel, year, cyl])
new_data = new_data.astype('float')
new_data = new_data.reshape(1, -1)
prediction = model.predict(new_data)

st.write("### Purple Key POC - Vehicle horsepower prediction")
st.write("The predicted horsepower is: ", f"{prediction[0]:.2f}")