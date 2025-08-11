import streamlit as st
import joblib

# Load the trained model and vectorizer
classifier = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
st.set_page_config(page_title="Spam Detection App", page_icon="📧", layout="centered")

# Title and description
st.title("📧 Email Spam Detection App")
st.markdown("""
This application uses a machine learning model to classify whether a given message is **Spam** or **Ham** (not spam). 
Simply enter your message below, and the app will predict its category.
""")

# Input text box
st.markdown("### Enter your message:")
user_input = st.text_area("Type your message here...", height=150)

# Prediction logic
if st.button("Predict"):
    if user_input.strip():
        # Transform the input and make a prediction
        input_vector = vectorizer.transform([user_input])
        prediction = classifier.predict(input_vector)[0]
        prediction_label = "Spam" if prediction == 1 else "Ham"

        # Display the result
        st.markdown(f"### Prediction: **{prediction_label}**")
        if prediction_label == "Spam":
            st.error("⚠️ This message is likely to be spam.")
        else:
            st.success("✅ This message is not spam.")
    else:
        st.warning("Please enter a valid message to classify.")

# Footer
st.markdown("---")
