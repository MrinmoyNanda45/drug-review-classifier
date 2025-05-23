# ✅ Final Updated app.py with Confidence Threshold Handling & Best Model Integration

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from textblob import TextBlob
from datetime import datetime
import re
import nltk
import seaborn as sns
nltk.download('punkt')

# === Confidence Threshold for Prediction Certainty ===
CONFIDENCE_THRESHOLD = 83.0  # % threshold for valid prediction

# === Load Models ===
model = joblib.load("best_model_tuned.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("numeric_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Keywords and Side Effects ===
positive_keywords = ['effective', 'relief', 'improved', 'better', 'great', 'helped']
negative_keywords = ['pain', 'side effect', 'worse', 'anxious', 'bad', 'problem', 'suffering']

common_side_effects = ['nausea', 'headache', 'dizziness', 'fatigue', 'insomnia','diarrhea', 'constipation', 'rash', 'dry mouth', 'weight gain','anxiety', 'vomiting', 'sweating', 'tremor', 'blurred vision']

# === Detailed Side Effect Metadata ===
side_effect_info = {
    "headache": {"type": "Neurological", "emoji": "💢", "color": "#cce5ff"},
    "nausea": {"type": "Digestive", "emoji": "🤢", "color": "#d4edda"},
    "dizziness": {"type": "Neurological", "emoji": "🌀", "color": "#cce5ff"},
    "fatigue": {"type": "General", "emoji": "😴", "color": "#f3d9fa"},
    "insomnia": {"type": "Neurological", "emoji": "🌙", "color": "#cce5ff"},
    "anxiety": {"type": "Neurological", "emoji": "😰", "color": "#cce5ff"},
    "depression": {"type": "Neurological", "emoji": "😔", "color": "#cce5ff"},
    "vomiting": {"type": "Digestive", "emoji": "🤮", "color": "#d4edda"},
    "dry mouth": {"type": "General", "emoji": "💧", "color": "#f3d9fa"},
    "constipation": {"type": "Digestive", "emoji": "💩", "color": "#d4edda"},
    "diarrhea": {"type": "Digestive", "emoji": "🚽", "color": "#d4edda"},
    "blurred vision": {"type": "Neurological", "emoji": "👓", "color": "#cce5ff"},
    "tremor": {"type": "Neurological", "emoji": "🫨", "color": "#cce5ff"},
    "sweating": {"type": "General", "emoji": "💦", "color": "#f3d9fa"},
    "weight gain": {"type": "General", "emoji": "⚖️", "color": "#f3d9fa"},
    "rash": {"type": "General", "emoji": "🌡️", "color": "#f3d9fa"},
    "tired": {"type": "General", "emoji": "😪", "color": "#f3d9fa"},
    "back pain": {"type": "Musculoskeletal", "emoji": "🚶‍♂️", "color": "#ffeeba"},
    "pain": {"type": "General", "emoji": "🔥", "color": "#f3d9fa"},
    "upset stomach": {"type": "Digestive", "emoji": "🤒", "color": "#d4edda"},
    "muscle pain": {"type": "Musculoskeletal", "emoji": "🏋️", "color": "#ffeeba"},
    "irritated skin": {"type": "Skin", "emoji": "🧴", "color": "#fff3cd"},
    "lightheaded": {"type": "Neurological", "emoji": "🌫️", "color": "#cce5ff"},
    "memory loss": {"type": "Neurological", "emoji": "🧠", "color": "#cce5ff"},
    "dry eyes": {"type": "General", "emoji": "👁️", "color": "#f3d9fa"},
    "cramps": {"type": "Musculoskeletal", "emoji": "🩻", "color": "#ffeeba"},
    "shortness of breath": {"type": "Respiratory", "emoji": "🫁", "color": "#e2e3e5"},
    "infection": {"type": "General", "emoji": "🦠", "color": "#f3d9fa"},
    "palpitations": {"type": "Cardiac", "emoji": "❤️", "color": "#f8d7da"},
    "mood swings": {"type": "Neurological", "emoji": "🎭", "color": "#cce5ff"},
    "sleepiness": {"type": "Neurological", "emoji": "😪", "color": "#f3d9fa"},
    "confusion": {"type": "Neurological", "emoji": "😵", "color": "#cce5ff"},
    "itching": {"type": "Skin", "emoji": "🪳", "color": "#fff3cd"},
    "heartburn": {"type": "Digestive", "emoji": "🔥", "color": "#d4edda"},
    "dry skin": {"type": "Skin", "emoji": "🧴", "color": "#fff3cd"},
    "irritability": {"type": "Neurological", "emoji": "😡", "color": "#cce5ff"},
    "numbness": {"type": "Neurological", "emoji": "🧊", "color": "#cce5ff"},
    "joint pain": {"type": "Musculoskeletal", "emoji": "🦴", "color": "#ffeeba"},
    "cold hands": {"type": "General", "emoji": "🧤", "color": "#f3d9fa"},
    "difficulty sleeping": {"type": "Neurological", "emoji": "🛌", "color": "#cce5ff"},
    "skin peeling": {"type": "Skin", "emoji": "🫳", "color": "#fff3cd"},
    "blurred thinking": {"type": "Neurological", "emoji": "🧠", "color": "#cce5ff"},
    "increased appetite": {"type": "General", "emoji": "🍽️", "color": "#f3d9fa"},
    "restlessness": {"type": "Neurological", "emoji": "🔄", "color": "#cce5ff"},
    "hair loss": {"type": "General", "emoji": "🧑‍🦲", "color": "#f3d9fa"}
}


# === Helpers ===
def highlight_keywords(text):
    for word in positive_keywords:
        text = re.sub(fr"\b({word})\b", r"<span style='color:green; font-weight:bold;'>\1</span>", text, flags=re.IGNORECASE)
    for word in negative_keywords:
        text = re.sub(fr"\b({word})\b", r"<span style='color:red; font-weight:bold;'>\1</span>", text, flags=re.IGNORECASE)
    return text

def get_sentiment_label(score):
    if score > 0.6:
        return "😍 Very Positive"
    elif score > 0.3:
        return "😊 Positive"
    elif score < -0.6:
        return "💔 Very Negative"
    elif score < -0.3:
        return "☹️ Negative"
    else:
        return "😐 Neutral"

def pretty_label(raw_label):
    return {
        "Depression": "🧠 Depression",
        "Diabetes, Type 2": "💉 Type 2 Diabetes",
        "High Blood Pressure": "💓 High Blood Pressure"
    }.get(raw_label, raw_label)

# === UI Setup ===
st.set_page_config(page_title="Drug Review Classifier", page_icon="💊", layout="centered")
st.markdown("<h1 style='text-align:center;'>💊 Drug Review Condition Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>🩺 Detect Condition, 💥 Side Effects and 🤖 Analyze Sentiment </h4><hr>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Single Review", "📁 Batch Upload"])

# === SINGLE REVIEW TAB ===
with tab1:
    st.subheader("📝 Enter Patient Review")
    with st.form("review_form"):
        review = st.text_area("Type your drug experience here:", height=150)
        col1, col2 = st.columns(2)
        with col1:
            rating = st.slider("⭐ Drug Rating", 1, 10, value=7)
        with col2:
            useful_count = st.number_input("👍 Helpful Votes", min_value=0, value=10)
        submit = st.form_submit_button("🔍 Predict")

    if submit and review.strip():
        with st.spinner("Analyzing..."):
            review_clean = re.sub(r"[^\w\s]", " ", review.lower())  # ✅ lowercase + punctuation cleanup
            sentiment_score = TextBlob(review_clean).sentiment.polarity
            review_length = len(review_clean.split())
            sentiment_label = get_sentiment_label(sentiment_score)

            X_text = tfidf_vectorizer.transform([review_clean])
            X_num = scaler.transform([[rating, useful_count, sentiment_score, review_length]])
            X_input = hstack([X_text, X_num])

            proba = model.predict_proba(X_input)[0]
            pred = np.argmax(proba)
            confidence = proba[pred] * 100
            is_uncertain = confidence < CONFIDENCE_THRESHOLD

            condition = label_encoder.inverse_transform([pred])[0] if not is_uncertain else "Uncertain / Out of Scope"
            sorted_idx = np.argsort(proba)[::-1]
            sorted_probs = proba[sorted_idx]
            sorted_labels = [pretty_label(label_encoder.classes_[i]) for i in sorted_idx]

        if is_uncertain:
            st.warning("⚠️ The model is not confident enough to classify this review under Depression, Type 2 Diabetes, or High Blood Pressure.")
            st.info(f"🔍 Maximum Confidence: **{confidence:.2f}%**")
        else:
            st.success(f"🩺 Predicted Condition: **{pretty_label(condition)}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")

        # Confidence bar
        st.markdown("### 📊 Model Confidence")
        fig, ax = plt.subplots(figsize=(6, 2.8))
        colors = sns.color_palette("coolwarm", len(sorted_probs))
        bars = ax.barh(sorted_labels, sorted_probs * 100, color=colors)
        for bar in bars:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}%", va='center')
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.set_xlabel("Confidence (%)")
        st.pyplot(fig)

        # Highlights
        st.markdown("### 🔍 Keyword Highlights")
        st.markdown(highlight_keywords(review), unsafe_allow_html=True)

        # Side effects
        detected = []
        for effect in common_side_effects:
            pattern = re.compile(re.escape(effect), flags=re.IGNORECASE)
            if pattern.search(review_clean):
                detected.append(effect)
        st.markdown("### 💥 Detected Side Effects")
        if detected:
            tags_html = ""
            for effect in detected:
                info = side_effect_info.get(effect, {"type": "General", "emoji": "❓", "color": "#e2e3e5"})
                tags_html += f"<span title='{info['type']} side effect' style='background-color:{info['color']};padding:6px 10px;border-radius:12px;margin:4px;display:inline-block;font-weight:600;color:#003366;font-size:14px;cursor:default;'>{info['emoji']} {effect}</span>"
            st.markdown(f"<div style='margin-top:10px;'>{tags_html}</div>", unsafe_allow_html=True)
        else:
            st.info("No common side effects detected.")

        # Summary
        st.markdown("### 📋 Summary")
        st.markdown(f"- ⭐ Rating: `{rating}`")
        st.markdown(f"- 👍 Helpful Votes: `{useful_count}`")
        st.markdown(f"- 💬 Sentiment Score: `{round(sentiment_score, 3)}`")
        st.markdown(f"- 🎝️ Sentiment: `{sentiment_label}`")
        st.markdown(f"- 📝 Word Count: `{review_length}`")

        # Download result
        df_result = pd.DataFrame([{
            "Review": review,
            "Predicted Condition": condition,
            "Confidence (%)": round(confidence, 2),
            "Rating": rating,
            "Helpful Votes": useful_count,
            "Sentiment": sentiment_score,
            "Sentiment Label": sentiment_label,
            "Review Length": review_length,
            "Side Effects": ", ".join(detected)
        }])
        st.download_button("⬇️ Download Result as CSV", data=df_result.to_csv(index=False).encode("utf-8"),
                           file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")




# === BATCH UPLOAD TAB ===
with tab2:
    st.subheader("📁 Upload CSV with Reviews")
    st.markdown("Ensure your file contains a `review` column.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("CSV must include a 'review' column.")
        else:
            with st.spinner("Processing batch..."):
                reviews = df["review"].fillna("").astype(str)
                sentiments = reviews.apply(lambda r: TextBlob(r.lower()).sentiment.polarity)
                lengths = reviews.apply(lambda r: len(r.split()))
                ratings = df.get("rating", pd.Series([7]*len(reviews)))
                votes = df.get("usefulCount", pd.Series([5]*len(reviews)))

                X_texts = tfidf_vectorizer.transform(reviews)
                numeric = np.array([ratings, votes, sentiments, lengths]).T
                X_num = scaler.transform(numeric)
                X_all = hstack([X_texts, X_num])

                preds = model.predict(X_all)
                probas = model.predict_proba(X_all)
                labels = label_encoder.inverse_transform(preds)
                confidences = [max(p)*100 for p in probas]

                final_labels = []
                for label, conf in zip(labels, confidences):
                    if conf < CONFIDENCE_THRESHOLD:
                        final_labels.append("Uncertain / Out of Scope")
                    else:
                        final_labels.append(label)

                df["Predicted Condition"] = final_labels
                df["Confidence (%)"] = [round(c, 2) for c in confidences]
                df["Sentiment"] = sentiments
                df["Review Length"] = lengths
                df["Sentiment Label"] = sentiments.apply(get_sentiment_label)

            st.success("✅ Batch prediction complete.")
            st.dataframe(df[["review", "Predicted Condition", "Confidence (%)", "Sentiment Label"]])
            st.download_button("⬇️ Download Full Results", df.to_csv(index=False).encode("utf-8"),
                               file_name="batch_predictions.csv", mime="text/csv")


# === Footer ===
st.markdown("---")
st.caption("🧪 Project: Patient's Condition Classification Using Drug Reviews | Built with 💻 Streamlit & NLP")
