import streamlit as st
import pickle
import numpy as np

# Try loading the model safely
try:
    pipe = pickle.load(open(r'C:\Users\Lenovo\laptop-price-predictor\pipe.pkl', 'rb'))
except ModuleNotFoundError as e:
    pipe = None
    st.error("⚠️ Required library not found. Please install it: `pip install xgboost`")

# Try loading dataframe
try:
    df = pickle.load(open(r'C:\Users\Lenovo\laptop-price-predictor\df.pkl', 'rb'))
except Exception as e:
    df = None
    st.error("⚠️ Data file could not be loaded. Please check your path to df.pkl")

st.title("💻 Laptop Price Predictor")

if df is not None:
    # Brand
    company = st.selectbox('Brand', df['Company'].unique())

    # Type of laptop
    laptop_type = st.selectbox('Type', df['TypeName'].unique())

    # RAM
    ram = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])

    # Weight
    weight = st.number_input('Weight of the Laptop (in kg)')

    # Touchscreen
    touchscreen = st.selectbox('Touchscreen', ['No','Yes'])
    touchscreen = 1 if touchscreen == 'Yes' else 0

    # IPS
    ips = st.selectbox('IPS', ['No','Yes'])
    ips = 1 if ips == 'Yes' else 0

    # Screen size
    screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)

    # Resolution
    resolution = st.selectbox('Screen Resolution', 
                              ['1920x1080','1366x768','1600x900','3840x2160','3200x1800',
                               '2880x1800','2560x1600','2560x1440','2304x1440'])
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    # CPU
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())

    # HDD
    hdd = st.selectbox('HDD (in GB)', [0,128,256,512,1024,2048])

    # SSD
    ssd = st.selectbox('SSD (in GB)', [0,8,128,256,512,1024])

    # GPU
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())

    # OS
    os = st.selectbox('OS', df['os'].unique())

    # Prediction button
    if st.button('Predict Price'):
        if pipe is not None:
            try:
                # Prepare input array
                query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi,
                                  cpu, hdd, ssd, gpu, os]).reshape(1, 12)
                
                # Predict and show result
                price = int(np.exp(pipe.predict(query)[0]))
                st.success(f"💰 The predicted price of this configuration is ₹{price}")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
        else:
            st.error("⚠️ Model is not loaded. Please install `xgboost` and restart.")
else:
    st.error("⚠️ Data not loaded. Cannot select features.")
