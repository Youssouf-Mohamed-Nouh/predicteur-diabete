import streamlit as st
import pandas as pd
import joblib
import subprocess 
import html
import re
from datetime import datetime
import platform
import os
st.set_page_config(page_title="Prédicteur Diabète  - Youssouf",page_icon='icon.jpg',layout='wide')
@st.cache_resource
def load_components():
    return(
        joblib.load('modelGDB.pkl'),
        joblib.load('scaler.pkl'),
        joblib.load('features.pkl')
        )
model,scaler,features = load_components()

# Header
st.markdown("""
            <div style=" background-color:#a6ebaa;paddind:20px;border-radius:15px;
            margin-bottom:30px;box-shadow:0 4px 10px rgba(0,0,0,0.1);">
            <h1 style="color:#8e98ec;text-align:center;font-size:42px;">
            Prédicteur Diabète
            </h1>
            <h3 style="color:#8e98ec;text-align:center;font-size:35px;">
            Assistant IA Médical - par Youssouf 
            </h3>
            
            </div>
            """,unsafe_allow_html=True)

# Formulaire
st.markdown(""" 
            <h2 style="color:#8e98ec;text-align:center;font-size:35px;">
            🧾 Formulaire du patients</h2>""",unsafe_allow_html=True)

with st.form('form_donnees'):
    col1,col2 = st.columns(2)
    with col1:
        prenom = st.text_input('👥 Votre Nom complet:').strip()
        age = st.slider('🎂 Age',min_value=18,max_value=100,value=30)
        preg = st.slider('👩‍👦 pregnancie(Nombre de grossesse)',0,20,1)
        Gluc = st.slider('🍭 Glucose(Taux de glucose sangain)',40,200,100)
        Blood = st.slider('📈 Bloodpressure(pression artérielle diastolique)',30,140,70)
    with col2:
        Skint = st.slider('🙇🏽‍ Skinthickess(Épaisseur du pli cutané)',0,100,20)
        insulin = st.slider('💊 insulin(Taux d\'insuline)',0,900,80)
        BMI =  st.slider('⚖️ BMI(Indeice de masse)',10,60,25)
        Diabete = st.slider('🩺 Diabetespedigreefunction(score fontion de prédispotion)',0.0,0.2,0.5,step=0.01)
    submit = st.form_submit_button('🔍 prédire le risque')
    

if submit:
    if not prenom:
        st.warning('Veuillez renseigner votre nom complet!')
        
    else:
        new_data = pd.DataFrame([[preg, Gluc, Blood, Skint, insulin, BMI, Diabete, age]],
                        columns=features)

        scaled_data = scaler.transform(new_data)
        prediction = model.predict(scaled_data)[0]
        proba = model.predict_proba(scaled_data)[0][1]
        st.markdown('------')
        st.subheader(f'résultats pour :{prenom}')
        if prediction == 1:
            st.error(f'{prenom},vous êtes risque le Diabète')
        else:
            st.success(f'{prenom},aucun risque pour le Diabète')
        with st.expander('🧾 données saisies:'):
            st.dataframe(new_data)
            