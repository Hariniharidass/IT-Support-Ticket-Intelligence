"""Streamlit app for predicting IT support ticket categories."""

from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path("models/ticket_classifier.joblib")
LOW_CONFIDENCE_THRESHOLD = 0.50


@st.cache_resource
def load_model():
    """Load the trained pipeline once and reuse it between predictions."""
    return joblib.load(MODEL_PATH)


st.title("IT Support Ticket Intelligence")
st.write("Enter a support ticket description to predict its category.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.training` first.")
    st.stop()

model = load_model()

st.write("**Available categories:** " + ", ".join(model.classes_))

ticket_text = st.text_area(
    "Ticket description",
    placeholder="Example: I cannot access the shared finance folder.",
    height=150,
)

if st.button("Predict category"):
    if ticket_text.strip():
        prediction = model.predict([ticket_text])[0]
        probabilities = model.predict_proba([ticket_text])[0]
        confidence = probabilities.max()

        st.success(f"Predicted category: {prediction}")
        st.write(f"Confidence: {confidence:.1%}")

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning("Low-confidence prediction — please review it manually.")
    else:
        st.warning("Enter a ticket description before predicting.")

st.caption("This learning project should not replace human review.")
