import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry

st.set_page_config(
    page_title="CFM Immobilier — Estimateur de Prix",
    page_icon="🏠",
    layout="wide"
)

# ── CSS personnalisé ──────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Outfit:wght@300;400;500&display=swap" rel="stylesheet">

<style>
  /* Police globale */
  html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
  }

  /* Header CFM */
  .cfm-header {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 0.5px solid #D3D1C7;
    margin-bottom: 2rem;
  }
  .cfm-logo-mark {
    width: 54px;
    height: 54px;
    background: #D85A30;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .cfm-logo-block {
    display: flex;
    flex-direction: column;
    gap: 0;
  }
  .cfm-logo-cfm {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #2C2C2A;
    letter-spacing: .05em;
    line-height: .95;
  }
  .cfm-logo-immo {
    font-family: 'Cormorant Garamond', serif;
    font-size: .95rem;
    font-weight: 300;
    color: #D85A30;
    letter-spacing: .28em;
    text-transform: uppercase;
    line-height: 1.4;
  }
  .cfm-sep {
    width: 1px;
    height: 52px;
    background: #D3D1C7;
    flex-shrink: 0;
    margin: 0 .25rem;
  }
  .cfm-tagline-main {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 400;
    font-style: italic;
    color: #2C2C2A;
    line-height: 1.3;
    display: block;
  }
  .cfm-tagline-sub {
    font-size: 12px;
    font-weight: 300;
    color: #5F5E5A;
    letter-spacing: .01em;
    display: block;
    margin-top: 4px;
  }

  /* Masquer le header Streamlit par défaut */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  /* Résultat */
  .cfm-result {
    background: #E1F5EE;
    border: 0.5px solid #5DCAA5;
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
  }
  .cfm-result-eyebrow {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .15em;
    color: #0F6E56;
    font-weight: 500;
  }
  .cfm-result-price {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 600;
    color: #085041;
    line-height: 1;
    margin: 6px 0 4px;
    letter-spacing: .02em;
  }
  .cfm-result-sub {
    font-size: 12px;
    color: #0F6E56;
    font-weight: 300;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="cfm-header">
  <div class="cfm-logo-mark">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.6">
      <path d="M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H4a1 1 0 01-1-1V9.5z"/>
      <path d="M9 21V12h6v9"/>
    </svg>
  </div>
  <div class="cfm-logo-block">
    <span class="cfm-logo-cfm">CFM</span>
    <span class="cfm-logo-immo">Immobilier</span>
  </div>
  <div class="cfm-sep"></div>
  <div>
    <span class="cfm-tagline-main">Renseignez les caractéristiques de votre bien</span>
    <span class="cfm-tagline-sub">et obtenez une estimation instantanée.</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Connexion Snowflake ───────────────────────────────────────
@st.cache_resource
def get_session():
    return get_active_session()

@st.cache_resource
def load_model():
    session = get_session()
    reg = Registry(
        session=session,
        database_name="HOUSES_PRICES_DB",
        schema_name="ML_SCHEMA"
    )
    return reg.get_model("HOUSE_PRICE_PREDICTOR").version("V1")

session = get_session()

# ── Formulaire ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Dimensions**")
    area = st.number_input(
        "Surface habitable (m²)",
        min_value=10, max_value=1000,
        value=150, step=1,
        help="Superficie totale du bien"
    )
    if area < 33 or area > 324:
        st.warning("Hors plage d'entraînement (33–324 m²) — estimation indicative.")
    bedrooms  = st.slider("Chambres",        1, 6, 3)
    bathrooms = st.slider("Salles de bain",  1, 4, 2)
    stories   = st.slider("Étages",          1, 4, 2)
    parking   = st.slider("Places de parking", 0, 4, 1)

with col2:
    st.markdown("**Caractéristiques**")
    mainroad  = st.radio("Route principale",  ["Oui", "Non"], horizontal=True)
    guestroom = st.radio("Chambre d'amis",    ["Oui", "Non"], horizontal=True, index=1)
    basement  = st.radio("Sous-sol",          ["Oui", "Non"], horizontal=True, index=1)
    prefarea  = st.radio("Zone privilégiée",  ["Oui", "Non"], horizontal=True, index=1)

with col3:
    st.markdown("**Équipements**")
    airconditioning  = st.radio("Climatisation",        ["Oui", "Non"], horizontal=True)
    hotwaterheating  = st.radio("Chauffage eau chaude", ["Oui", "Non"], horizontal=True, index=1)
    furnishingstatus = st.selectbox(
        "Ameublement",
        ["Meublé", "Semi-meublé", "Non meublé"],
        index=1
    )

st.divider()

# ── Encodage ─────────────────────────────────────────────────
def yn(val): return 1 if val == "Oui" else 0

furnishing_map = {"Meublé": 2, "Semi-meublé": 1, "Non meublé": 0}

input_data = pd.DataFrame([{
    "AREA"            : area,
    "BEDROOMS"        : bedrooms,
    "BATHROOMS"       : bathrooms,
    "STORIES"         : stories,
    "MAINROAD"        : yn(mainroad),
    "GUESTROOM"       : yn(guestroom),
    "BASEMENT"        : yn(basement),
    "HOTWATERHEATING" : yn(hotwaterheating),
    "AIRCONDITIONING" : yn(airconditioning),
    "PARKING"         : parking,
    "PREFAREA"        : yn(prefarea),
    "FURNISHINGSTATUS": furnishing_map[furnishingstatus]
}])

# ── Prédiction ────────────────────────────────────────────────
col_btn, col_result = st.columns([1, 2])

with col_btn:
    predict_btn = st.button("Estimer mon bien →", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Calcul en cours..."):
        try:
            model_mv   = load_model()
            prediction = model_mv.run(input_data, function_name="predict")
            pred_value = int(prediction.iloc[0, 0])
            price_per_m2 = pred_value // area

            with col_result:
                st.markdown(f"""
                <div class="cfm-result">
                  <div class="cfm-result-eyebrow">Estimation CFM Immobilier</div>
                  <div class="cfm-result-price">{pred_value:,} €</div>
                  <div class="cfm-result-sub">
                    Soit environ {price_per_m2:,} €/m² &nbsp;—&nbsp; estimation indicative
                  </div>
                </div>
                """.replace(",", "\u202f"), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur : {e}")
            st.info("Vérifiez que le modèle HOUSE_PRICE_PREDICTOR V1 est bien dans le registry.")

# ── Données envoyées au modèle ────────────────────────────────
with st.expander("Voir les données envoyées au modèle"):
    st.dataframe(input_data, use_container_width=True)

# ── Historique ────────────────────────────────────────────────
st.divider()
st.markdown("#### Historique des inférences")

try:
    history = session.table("HOUSES_PRICES_DB.ML_SCHEMA.INFERENCE_RESULTS").to_pandas()
    st.dataframe(
        history[["AREA","BEDROOMS","BATHROOMS","AIRCONDITIONING",
                 "PREFAREA","PREDICTED_PRICE"]].head(20),
        use_container_width=True
    )
    st.caption(f"Total : {len(history)} prédictions stockées")
except Exception:
    st.info("Aucune donnée d'inférence disponible pour l'instant.")

