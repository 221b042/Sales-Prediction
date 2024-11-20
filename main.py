import numpy as np
import datetime as dt
import joblib
import streamlit as st
import matplotlib.pyplot as plt

model = joblib.load('bigmart_model')

current_year = dt.datetime.today().year

st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>Sales Prediction System</h1>", unsafe_allow_html=True)

st.sidebar.title("App Info")
st.sidebar.info("Use this app to predict sales using a machine learning model. Fill in the details and click on 'Predict'.")

st.markdown("### Enter the following details for prediction:")

col1, col2 = st.columns(2)
with col1:
    item_mrp = st.number_input("Item Price", min_value=0.0, format="%.2f")
    outlet_size = st.selectbox("Outlet Size", ['High', 'Medium', 'Small'])
    outlet_type = st.selectbox("Outlet Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
with col2:
    outlet_identifier = st.selectbox("Outlet Identifier", 
                                    ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 
                                    'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
    outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1900, max_value=current_year)

outlet_age = current_year - outlet_establishment_year

# Convert values to numeric
outlet_identifier_map = {
    "OUT010": 0, "OUT013": 1, "OUT017": 2, "OUT018": 3, "OUT019": 4, 
    "OUT027": 5, "OUT035": 6, "OUT045": 7, "OUT046": 8, "OUT049": 9
}
outlet_size_map = {"High": 0, "Medium": 1, "Small": 2}
outlet_type_map = {
    "Grocery Store": 0, "Supermarket Type1": 1, "Supermarket Type2": 2, "Supermarket Type3": 3
}

# Mapping the inputs to numeric
p2 = outlet_identifier_map[outlet_identifier]
p3 = outlet_size_map[outlet_size]
p4 = outlet_type_map[outlet_type]
p5 = outlet_age

if st.button("Predict"):
    with st.spinner("Predicting... Please wait..."):
        prediction_input = np.array([[item_mrp, p2, p3, p4, p5]])
        result = model.predict(prediction_input)

    st.markdown(f"<h3 style='color: #1E90FF;'>Predicted Sales Amount: {float(result):.2f}</h3>", unsafe_allow_html=True)
    st.write(f"The sales amount is between: {float(result) - 714.42:.2f} and {float(result) + 714.42:.2f}")
    
    fig, ax = plt.subplots()
    
    bar_positions = [0, 1, 2]  # Positions for each bar
    bar_labels = ["Lower Bound", "Predicted Sales", "Upper Bound"]
    bar_values = [float(result) - 714.42, float(result), float(result) + 714.42]
    bar_colors = ['#ff7f0e', '#1f77b4', '#d62728']

    ax.bar(bar_positions, bar_values, color=bar_colors, alpha=0.8)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels)
    
    ax.set_title("Predicted Sales Amount with Range")
    ax.set_ylabel("Sales Amount")
    ax.grid(visible=True, linestyle='--', alpha=0.5)

    st.pyplot(fig)
