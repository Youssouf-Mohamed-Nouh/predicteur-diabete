import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import tempfile
import warnings
warnings.filterwarnings('ignore')
# configration de la page
st.set_page_config(page_title='Mon Assistant Santé : Évaluation de risque de diabète',
                   page_icon='🩺',
                   layout='wide',
                   initial_sidebar_state='expanded')

@st.cache_resource
def load_composents():
    try:
        model = joblib.load('modelGDB.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        return model,scaler,features
    except FileNotFoundError as e:
        st.error(f'Oups ! il semble qu\'un fichier soit manquant :{e}')
        st.info('Assurez-vous que tous les fichiers du modele sont présent dans le dossier')
        st.stop()
    except Exception as e:
        st.error(f'Une erreur c\'est produite lors de chargement :{e}')
        st.info('Veuillez vérifier l\'ntégrité des fichiers du moele')
        st.stop()
model,scaler,features = load_composents()
st.markdown('''
 <style>
 .main-header{
    background: linear-gradient(135deg,#4CAF50 0%,#2196F3 100%);
    padding:2.5rem;
    border-radius:30px;  
    margin-bottom:2rem;
    box-shadow:0 10px 40px rgba(0,0,0,0.1);
    color:white;
    text-align:center;                      
     }
 </style>           
''',unsafe_allow_html=True)
# en tete
st.markdown('''
            <div class='main-header'>
            <h1>🩺 Mon Assistant Santé</h1>
            <h3>Votre compagnon pour l'évaluation du risque diabète</h3>
            <p>Une approche bienveillante et scientifique pour mieux comprendre votre santé</p>
            </div>
            
''',unsafe_allow_html=True)
# message 
st.markdown('''
 <style>
 .welcom-message{
    background: linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%);
    padding:2rem;
    border-radius:15px;  
    margin-bottom:2rem;
    border-left: 5px solid #4CAF50 ;       
     }
 </style>           
''',unsafe_allow_html=True)
st.markdown('''
            <div class='welcom-message'>
            <h1>👋 Bonjour et bienvenue !</h1>
            <p> Je suis votre assistant santé numérique , conçu pour vous aider à évaluer votre 
            risque de diabète de manière simple et accessible.
            Ensemble,nous allons analyser  quelquels information de base mieux comprendre votre profil
            de santé.</p>
            <p><strong> Rassurez-vous ! </strong> : cet outil est là pour vous accompagner,
            pas pour inquiéter.
            Il s'agit d'une première approche qui vous aidera à dialoguer avec votre médecin</p>
            </div>
            
''',unsafe_allow_html=True)


# information rassurant
# message 
st.markdown('''
 <style>
 .friendly-info{
    background: #cce6ff;
    padding:2rem;
    border-radius:15px;  
    border-left: 5px solid #2196F3;
    margin : 1.5rem 0;
                 
     }
 .encouragement{
     background: #cce6ff;
    padding:2rem;
    border-radius:15px;  
    border-left: 5px solid #2196F3;
    margin : 1.5rem 0;
                 
     }
 </style>           
''',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('🤖 À propos de votre assistant')
    st.markdown('''
                <div class='friendly-info'>
                <h4>Comment je fonctionne ?</h4>
                <p>• j'utilise un modèle entrainé sur des meilleur de cas </p>
                <p>• Ma précision est 90% </p>
                <p>• J'ai été mis à en Janvier 2025 </p>
                <p>• Je respecte votre vie privée </p>
                </div>
    ''',unsafe_allow_html=True)
    st.markdown('📢 Rappel Importante')
    st.markdown('''
                <div class='encouragement'>
                <p><strong> Gartez en tête :</strong></p>
                <p>• ⚒️ Je suis un outil d'aide,pas un diagnostic médical</p>
                <p>• 👨‍⚕️ 👩‍⚕️ Votre médecin reste votre meilleur allié</p>
                <p>• 🩺 Prendre soin de sa santé ,c'est un acte d'amour envers soi</p>
                
                </div>
    ''',unsafe_allow_html=True)

# form
st.markdown('📝Parlez-moi de vous ')
st.markdown('*Prenez votre temps pour remplir ces informations .chaque détail compte pour une évaluation précise.*')
with st.form('form_evaluation'):
    # section information personnelle
    st.markdown('### 👥 Qui êtes-vous ?')
    col1,col2 = st.columns(2)
    with col1:
        prenom = st.text_input(
            '📝 Votre nom complet',
            placeholder='Ex: Youssouf Mohamed',
            help='Ceci nous aide à personnaliser votre rapport').strip()
        age = st.slider('📅 Votre Âge',
                        min_value=18,max_value=70,value=25,step=1,
                        help='L\'âge influence les facteurs de risque diabètique')
        preg = st.slider('👶 Nombre de grossesse(si applicable)',
                        min_value=0,max_value=13,value=6,step=1,
                        help='Les grossessesnpeuvent influencer de risque diabètique')
    
    with col2:
        st.markdown('🤝 Quelques conseil')
        st.info('''
                *Avant de commencer*:
                - Ayez vos dernière analyses sous la main
                - Soyez honnête dans vos réponses
                - N'hésitez pas à estimer si vous n'êtes pas sûr(e)
                ''')
    st.markdown('----')
    # section medicale
    st.markdown('### 🩺 Vos données santé')
    st.markdown('*Si vous n\'avez pas de données récentes ,donnez votre meilleure estimation*')
    col1,col2 = st.columns(2)
    with col1:
        Gluc = st.slider('🍯 Glucose sanguin (mg/dl)',min_value=20,max_value=300,value=40,
                         help='Mesure généralement prise à jeun. Normal:70-100 mg/dl ')
        Blood = st.slider('ﮩ٨ـﮩﮩ٨ـ♡ﮩ٨ـﮩﮩ٨ـ Tention artiélle (systolique)',
                          min_value=30,max_value=300,value=80,step=1,
                          help='Le chiffre du haut de votre tension . Normal moins de 120')
        skint = st.slider('📏Épaisseur du pli cutané(mm)',
                          min_value=10,max_value=50,value=30,step=1,
                          help='Mesure généralement prise au niveau du triceps')
    with col2:
        insulin = st.slider('💊 Insuline (μUl/mL)',
                            min_value=10,max_value=300,value=100,step=1,
                            help='Taux d\'insuline dans le sang. Normal :2,6-24,9')
        BMI = st.slider('⚖️ Indice de masse corporelle(IMC)',min_value=14,max_value=100,value=10,step=1,
                        help='calculé avec votre poids et taille. Normal :18,5-24,9')
        Diabete = st.slider('🧬 Antécédents familiaux de diabète',
                            min_value=0.0,max_value=2.0,value=1.0,step=0.01,
                            help='Score basé sur vos antécédents familiaux (0 = aucun , 2 = nombreux)')
    st.markdown('----')
            
    
    # avant soumission
    st.markdown('''
             <div class="encouragement">
             <p>⭐ <strong> Vous y êtes presque !</strong></p>
             <p>En cliquant sur le bouton ci-dessous , vous obtiendrez un évaluation personnalisée de votre profil de santé.
             Souvenez-vous:quelle que soit l'évaluation , vous avez le pouvoir d'agir positivement sur votre santé</p>
             </div>   
    ''',unsafe_allow_html=True)    
    submit =st.form_submit_button(
        'Decouvrir mon profil santé',
        type='primary',
        use_container_width=True)

# traitement
st.markdown('''
 <style>
 .risk-high{
    background: linear-gradient(135deg,#ff7675,#fd79a8);
    color:white;                            
    padding:2rem;
    border-radius:15px; 
    text-align:center;
    margin : 1.5rem 0;
    box-shadow:0 8px 30px rgba(255,118,117,0.3);
                 
     }
 .risk-low{
     background: linear-gradient(135deg,#00b894,#00cec9);
     color:white;                            
     padding:2rem;
     border-radius:15px;
     text-align:center;
     width:fit-content;
     margin : 1.5rem 0;
     box-shadow:0 8px 30px rgba(0,184,148,0.3);
     }                           
 </style>           
''',unsafe_allow_html=True)
if submit:
    if not prenom:
        st.warning("Pour personnaliser votre expérience, pourriez-vous nous dire comment vous appeler ?")
    else:
        new_data = pd.DataFrame([[preg, Gluc, Blood, skint, insulin, BMI, Diabete, age]], columns=features)
        
        with st.spinner("Analyse de votre profil en cours..."):
            scaled_data = scaler.transform(new_data)
            prediction = int(model.predict(scaled_data)[0])
            proba = model.predict_proba(scaled_data)[0]
            risk_percentage = proba[1] * 100

        st.markdown("---")
        st.markdown(f"### 🎯 Votre santé, {prenom}")
        col1, col2 = st.columns([2, 2])

        with col1:
            if prediction == 1:
                st.markdown(f"""
                    <div class='risk-high'>
                    <h3>🚨 Attention recommandée</h3>
                    <p><strong>{prenom}</strong>, votre profil suggère un risque plus élevé</p>
                    <p>Probabilité estimée : {risk_percentage:.1f}%</p>
                    <p><em>Mais ne vous inquiétez pas, c'est le moment parfait pour agir !</em></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='risk-low'>
                    <h3>🙌 Excellente nouvelle !</h3>
                    <p><strong>{prenom}</strong>, votre profil suggère un risque plus faible</p>
                    <p>Probabilité estimée : {risk_percentage:.1f}%</p>
                    <p><em>Continuez sur cette belle lancée !</em></p>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📊 Vos indicateurs de santé")
            health_indicators = {
                'Indicateur de santé': ['Taux de glucose', 'IMC', 'Âge', 'Tension artérielle', 'Insuline'],
                'Votre valeur': [Gluc, BMI, age, Blood, insulin],
                'Statut': []
            }

            for indicator, value in zip(health_indicators['Indicateur de santé'], health_indicators['Votre valeur']):
                if indicator == 'Taux de glucose':
                    status = '✅ Excellent' if value <= 100 else '⚠️ À surveiller' if value <= 125 else '🔴 Attention'
                elif indicator == 'IMC':
                    status = '✅ Normal' if 18.5 <= value <= 24.9 else '⚠️ Surpoids' if value <= 29.9 else '🔴 Obésité'
                elif indicator == 'Âge':
                    status = '💪 Jeune' if value <= 35 else '🧑‍💼 Adulte' if value <= 55 else '👴 Senior'
                elif indicator == 'Tension artérielle':
                    status = '✅ OK' if value <= 120 else '⚠️ Limite' if value <= 139 else '🔴 Élevée'
                else:  # Insuline
                    status = '✅ Normal' if value <= 25 else '⚠️ Modérée' if value <= 100 else '🔴 Élevée'
                health_indicators['Statut'].append(status)

            st.dataframe(pd.DataFrame(health_indicators), use_container_width=True, hide_index=True)

        with st.expander("🔍 Décryptage complet de votre profil santé"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 📋 Récapitulatif")
                df = new_data.copy()
                df.columns = ['Grossesses', 'Glucose', 'Tension', 'Pli cutané', 'Insuline', 'IMC', 'H. familial', 'Âge']
                st.dataframe(df.style.format({
                    'Glucose': '{:.0f} mg/dL',
                    'Tension': '{:.0f} mmHg',
                    'Pli cutané': '{:.0f} mm',
                    'Insuline': '{:.0f} μIU/mL',
                    'IMC': '{:.1f}',
                    'H. familial': '{:.2f}',
                    'Âge': '{:.0f} ans'
                }), use_container_width=True)

            with col_b:
                st.markdown("### 🎯 Analyse du risque")
                risk_data = pd.DataFrame({
                    "Niveau de risque": ["Faible", "Vigilance"],
                    "Probabilité": [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                    "Recommandation": ["Maintien", "Suivi médical"]
                })
                st.dataframe(risk_data, use_container_width=True, hide_index=True)

                st.markdown("### 💬 Mon analyse")
                if risk_percentage < 50:
                    st.success("🌟 Votre profil est rassurant. Continuez vos bonnes habitudes !")
                elif 50 <= risk_percentage < 70:
                    st.warning("⚠️ Vigilance recommandée. Quelques ajustements peuvent suffire.")
                else:
                    st.error("🚨 Risque élevé. Consultez un professionnel de santé rapidement.")

# Message de conclusion plus chaleureux
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">🩺 Votre Assistant Santé Personnel</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créé avec passion par <strong>Youssouf</strong> pour vous accompagner dans votre parcours santé
    </p>
    <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 1rem;">
        Version 2024 - Mis à jour régulièrement pour votre bien-être
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem;">
        <p style="font-size: 0.85em; color: #6c757d; font-style: italic;">
            ⚠️ Rappel important : Cet outil d'aide à la décision complète mais ne remplace jamais 
            l'expertise de votre médecin traitant
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
