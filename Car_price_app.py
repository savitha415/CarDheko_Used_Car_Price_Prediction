import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="CarDekho - Price Predictor", page_icon="🚗", layout="wide")

# ---------- CUSTOM CSS STYLING ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: #ffe5b4;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #d9534f;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #c9302c;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA AND MODEL ----------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_datas.csv")

@st.cache_resource
def load_model():
    with open("pipeline.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# ---------- LOGO / HEADER ----------
try:
    logo = Image.open("car_logo.png")
    st.image(logo, width=120)
except:
    st.title("🚗 CarDekho Price Prediction")

st.markdown("<h2 style='color:#d9534f;'>📊 Find the Best Selling Price for Your Car</h2>", unsafe_allow_html=True)
st.markdown("---")

# ---------- INPUT COLUMNS ----------
col1, col2, col3, col4 = st.columns(4)

# --- Column 1 ---
with col1:
    st.markdown("### 🔧 Vehicle Info")
    brand = st.selectbox("🏷️ Brand", df["Brand"].unique())
    model_name = st.selectbox("🚘 Model", df[df["Brand"] == brand]["model"].unique())
    model_year = st.selectbox("📅 Year", sorted(df["modelYear"].unique()))
    seats = st.selectbox("💺 Seating Capacity", sorted(df["Seats"].unique()))

# --- Column 2 ---
with col2:
    st.markdown("### ⚙️ Engine & Fuel")
    fuel_type = st.selectbox("⛽ Fuel Type", ['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'])
    transmission = st.selectbox("🔁 Transmission", ['Manual', 'Automatic'])
    mileage = st.number_input("📏 Mileage (km/l)", min_value=5, max_value=50, step=1)
    kms_driven = st.number_input("🧭 Kilometers Driven", min_value=100, max_value=200000, step=100)

# --- Column 3 ---
with col3:
    st.markdown("### 🏙️ City & Body")
    body_type = st.selectbox("🚙 Body Type", [
        'Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe',
        'Minivans', 'Convertibles', 'Hybrids', 'Wagon', 'Pickup Trucks'])
    city = st.selectbox("📍 Registered City", df["City"].unique())
    color = st.selectbox("🎨 Color", df["Color"].unique())
    owner = st.selectbox("🧑‍🔧 Previous Owners", [0, 1, 2, 3, 4, 5])

# --- Column 4 ---
with col4:
    st.markdown("### 🛡️ Insurance")
    insurance = st.selectbox("📝 Insurance Validity", [
        'Third Party insurance', 'Comprehensive',
        'Third Party', 'Zero Dep', '2', '1', 'Not Available'])

# ---------- PREDICT BUTTON ----------
st.markdown("---")
if st.button("🚀 Predict Price"):
    with st.spinner("🔍 Analyzing and Predicting Car Price..."):
        input_data = pd.DataFrame({
            'Fuel type': [fuel_type],
            'body type': [body_type],
            'transmission': [transmission],
            'ownerNo': [owner],
            'Brand': [brand],
            'model': [model_name],
            'modelYear': [model_year],
            'Insurance Validity': [insurance],
            'Kms Driven': [kms_driven],
            'Mileage': [mileage],
            'Seats': [seats],
            'Color': [color],
            'City': [city]
        })

        predicted_price = model.predict(input_data)[0]

        st.success("🎯 Price Prediction Complete!")

        # ---------- RESULT TABLE ----------
        st.markdown("### 💰 Estimated Car Details")
        result_df = pd.DataFrame({
            'Brand': [brand],
            'Model': [model_name],
            'Year': [model_year],
            'Fuel Type': [fuel_type],
            'Transmission': [transmission],
            'Mileage (km/l)': [mileage],
            'Kilometers Driven': [f"{kms_driven:,}"],
            'Seats': [seats],
            'Body Type': [body_type],
            'Color': [color],
            'City': [city],
            'Insurance': [insurance],
            'Previous Owners': [owner],
            'Estimated Price (Lakhs)': [f"₹ {predicted_price:.2f}"]
        })

        st.table(result_df)
