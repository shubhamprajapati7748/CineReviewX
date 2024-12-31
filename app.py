import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model 
import streamlit as st

## Loading the model 

try: 
    model = load_model("Artifacts/simple_rnn_model.h5")
except: 
    print("unable to load the model file")
    
## Loading the IMDB dataset word index 
word_index = imdb.get_word_index()

# Helper function to preprocess user input text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Default unknown words map to index 2
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    try:
        preprocessed_input = preprocess_text(review)
        prediction = model.predict(preprocessed_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        return sentiment, prediction[0][0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

## streamlit ap 
st.title("Movie Reviews Sentiment Analysis")
st.write("Enter a movie review to classifly it as Positive or negative")

## user input 
user_input = st.text_area("Movie Review ")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review to analyze")
    else: 
        sentiment, prediction_score = predict_sentiment(user_input)
        if sentiment:
            st.subheader(f"Sentiment: {sentiment}")
            st.write(f"Prediction Score: {prediction_score:.4f}")
            st.write("Based on the score above, we classified the sentiment as: ")
            if sentiment == "Positive":
                st.success("This review is positive!")
            else:
                st.success("This review is negative!")

else:
    st.info("Please enter a moview review to get the sentiment analysis.")


# Additional styling for a cleaner look
st.markdown("""
    <style>
        .css-ffhzg2 {
            font-size: 1.5em;
            color: #2c3e50;
        }
        .css-1n7v3bs {
            padding: 1rem;
            background-color: #ecf0f1;
            border-radius: 8px;
        }
        .stTextInput, .stTextArea {
            font-size: 1.2em;
        }
    </style>
""", unsafe_allow_html=True)