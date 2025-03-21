from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import json
import pickle
import numpy as np

with open('../models/Random_Forest_Bitterness.sav', 'rb') as file:
    model = pickle.load(file)

#with open('../data/processed/dic_cl.json','r', encoding='utf-8') as archivo:
#    dic_cl = json.load(archivo)
#with open('../data/processed/dic_el.json','r', encoding='utf-8') as archivo:
#    dic_el = json.load(archivo)
#with open('../data/processed/dic_et.json','r', encoding='utf-8') as archivo:
#    dic_et = json.load(archivo)
#with open('../data/processed/dic_er.json','r', encoding='utf-8') as archivo:
#    dic_er = json.load(archivo)
#with open('../data/processed/dic_cs.json','r', encoding='utf-8') as archivo:
#    dic_cs = json.load(archivo)

st.title("Predicción amargo cocimientos estándar")

val1 = st.slider("Número de latas de Extracto CO2", min_value = 3, max_value = 4, step = 1)
val2 = st.slider("Kilogramos de lúpulo Herkules", min_value = 35, max_value = 50, step = 1)
val3 = st.slider("% de alpha acidos lúpulo Herkules", min_value = 13.4, max_value = 15.5, step = 0.1)
val4 = st.slider("mg de catalizador", min_value = 1800.0, max_value = 2600.0, step = 10.0)
#val2 = st.selectbox(
#    "Nivel de Conocimiento del Empleado",
#    (dic_el.keys()),
#    index=None,
#    placeholder="Selecciona nivel de experiencia....",
#)
#val3 = st.selectbox(
#    "Tipo de Contrato",
#    (dic_et.keys()),
#    index=None,
#    placeholder="Selecciona tipo de empleo....",
#)
#val4 = st.slider("Porcentaje de Trabajo Remoto", min_value = 0, max_value = 100, step = 1)
#val5 = st.selectbox(
#    "Tamaño de la Compañia",
#    (dic_cs.keys()),
#    index=None,
#    placeholder="Selecciona el tamaño...",
#)

#val6 = st.selectbox(
#    "Pais de Residencia del Empleado",
#    (dic_er.keys()),
#    index=None,
#    placeholder="Selecciona el pais...",
#)

#val7 = st.selectbox(
#    "Pais de Origen de la Compañia",
#    (dic_cl.keys()),
#    index=None,
#    placeholder="Selecciona el pais...",
#)


if st.button("Predecir"):
    # Create input array and reshape for prediction
    X_pred = np.array([val1, val2, val3, val4]).reshape(1, -1)
    prediction = model.predict(X_pred)[0]
    st.write(f"Amargo por cocimiento predicho: {prediction:.2f} IBUs")
