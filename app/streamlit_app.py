
# STREAMLIT IN SNOWFLAKE — Application de prédiction de prix
# Déployer dans : Snowflake > Streamlit > + Streamlit App

import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry

# ── Configuration de la page ──────────────────────────────────
st.set_page_config(
    page_title=" Estimateur de Prix Immobilier",
    page_icon="🏠",
    layout="wide"
)

# ── Connexion Snowflake ───────────────────────────────────────
@st.cache_resource
def get_session():
    return get_active_session()

@st.cache_resource
def load_model():
    session = get_session()
    reg = Registry(
        session=session,
        database_name="HOUSE_PRICE_DB",
        schema_name="ML_SCHEMA"
    )
    return reg.get_model("HOUSE_PRICE_PREDICTOR").version("V1")

session = get_session()

# ── Header ────────────────────────────────────────────────────
st.title(" Estimateur de Prix Immobilier")
st.markdown("*Powered by Snowflake ML — Entrez les caractéristiques d'un bien pour obtenir une estimation*")
st.divider()

# ── Formulaire de saisie ──────────────────────────────────────
st.subheader(" Caractéristiques du bien")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("** Dimensions**")
    area      = st.number_input("Surface (m²)", min_value=500, max_value=20000,
                                 value=5000, step=100)
    bedrooms  = st.slider("Chambres", 1, 6, 3)
    bathrooms = st.slider("Salles de bain", 1, 4, 2)
    stories   = st.slider("Étages", 1, 4, 2)
    parking   = st.slider("Places de parking", 0, 4, 1)

with col2:
    st.markdown("** Caractéristiques**")
    mainroad  = st.radio("Route principale", ["Oui", "Non"], horizontal=True)
    guestroom = st.radio("Chambre d'amis",   ["Oui", "Non"], horizontal=True)
    basement  = st.radio("Sous-sol",         ["Oui", "Non"], horizontal=True)
    prefarea  = st.radio("Zone privilégiée", ["Oui", "Non"], horizontal=True)

with col3:
    st.markdown("** Équipements**")
    airconditioning   = st.radio("Climatisation",       ["Oui", "Non"], horizontal=True)
    hotwaterheating   = st.radio("Chauffage eau chaude",["Oui", "Non"], horizontal=True)
    furnishingstatus  = st.selectbox(
        "Ameublement",
        ["Meublé", "Semi-meublé", "Non meublé"]
    )

st.divider()

# ── Encodage des inputs ───────────────────────────────────────
def encode_yes_no(val):
    return 1 if val == "Oui" else 0

furnishing_map = {"Meublé": 2, "Semi-meublé": 1, "Non meublé": 0}

input_data = pd.DataFrame([{
    'AREA'            : area,
    'BEDROOMS'        : bedrooms,
    'BATHROOMS'       : bathrooms,
    'STORIES'         : stories,
    'MAINROAD'        : encode_yes_no(mainroad),
    'GUESTROOM'       : encode_yes_no(guestroom),
    'BASEMENT'        : encode_yes_no(basement),
    'HOTWATERHEATING' : encode_yes_no(hotwaterheating),
    'AIRCONDITIONING' : encode_yes_no(airconditioning),
    'PARKING'         : parking,
    'PREFAREA'        : encode_yes_no(prefarea),
    'FURNISHINGSTATUS': furnishing_map[furnishingstatus]
}])

# ── Bouton de prédiction ──────────────────────────────────────
col_btn, col_result = st.columns([1, 2])

with col_btn:
    predict_btn = st.button(" Estimer le prix", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Calcul en cours..."):
        try:
            model_mv = load_model()
            prediction = model_mv.run(input_data, function_name="predict")

            # Extraire la valeur
            pred_value = int(prediction.iloc[0, 0])

            with col_result:
                st.success(f"###  Prix estimé : **{pred_value:,} €**")
                st.caption(f"Basé sur : {area} m², {bedrooms} ch., {bathrooms} sdb., "
                           f"{'climatisé' if airconditioning == 'Oui' else 'non climatisé'}, "
                           f"zone {'privilégiée' if prefarea == 'Oui' else 'standard'}")

        except Exception as e:
            st.error(f"Erreur : {e}")
            st.info("Vérifiez que le modèle HOUSE_PRICE_PREDICTOR V1 est bien dans le registry.")

# ── Récapitulatif des données saisies ─────────────────────────
with st.expander(" Voir les données envoyées au modèle"):
    st.dataframe(input_data, use_container_width=True)

# ── Historique des prédictions ────────────────────────────────
st.divider()
st.subheader(" Historique des inférences")

try:
    history = session.table("HOUSE_PRICE_DB.ML_SCHEMA.INFERENCE_RESULTS").to_pandas()
    st.dataframe(
        history[['AREA','BEDROOMS','BATHROOMS','AIRCONDITIONING',
                 'PREFAREA','PREDICTED_PRICE']].head(20),
        use_container_width=True
    )
    st.caption(f"Total : {len(history)} prédictions stockées")
except Exception:
    st.info("Aucune donnée d'inférence disponible pour l'instant.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px'>"
    "MBAESG — Workshop Snowflake Data Engineering & ML</div>",
    unsafe_allow_html=True
)

