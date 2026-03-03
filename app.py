import streamlit as st
import joblib
import re
import string

# =============================
# Função de limpeza
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# =============================
# Configuração da página
# =============================
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Disaster Tweet Classifier")
st.markdown("Classifique tweets como **desastre real** ou **uso metafórico**.")

# =============================
# Carregar modelo
# =============================
@st.cache_resource
def load_model():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# =============================
# Input do usuário
# =============================
tweet = st.text_area("Digite um tweet:")

if st.button("Classificar"):
    if tweet.strip() == "":
        st.warning("Digite um tweet antes de classificar.")
    else:
        cleaned = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0][1]

        if prediction == 1:
            st.error(f"🚨 Desastre REAL detectado\n\nConfiança: {probability:.2%}")
        else:
            st.success(f"✅ Não é um desastre real\n\nConfiança: {1 - probability:.2%}")
