import streamlit as st
import pandas as pd
import re
import numpy as np
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
import zipfile

# Download the stopwords resource if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Extract the zip file if not already extracted
zip_path = 'best_model_lstm_2.zip'
extraction_path = 'extracted_model'

if not os.path.exists(extraction_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

# Load the trained model from the extracted folder
model_path = os.path.join(extraction_path, 'best_model_lstm_2') 
model = keras.models.load_model(model_path)

# Load stopwords
stpwds_en = set(stopwords.words('english')) 

# Text preprocessing function
def text_preprocessing(text):
    # Case folding
    text = text.lower()

    # Mention removal
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)

    # Hashtags removal
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)

    # Newline removal
    text = re.sub(r"\n", " ", text)

    # Whitespace removal
    text = text.strip()

    # URL removal
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Non-letter removal (such as emoticons, symbols, etc.)
    text = re.sub(r"[^A-Za-z\s']", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords removal
    tokens = [word for word in tokens if word not in stpwds_en]

    # Combining Tokens
    text = ' '.join(tokens)

    return text

# Function to check sentiment
def check_sentiment(inf_text):
    processed_text = text_preprocessing(inf_text)
    prediction = model.predict(np.array([[processed_text]]))  
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    predicted_class_index = np.argmax(prediction)
    return sentiment_classes[predicted_class_index]

def run():
    # Set the title of the web app
    st.title("Hotel Review Sentiment Analysis")
    st.write("---")
    
    # Banner
    st.image('sa.png')

    # Description of the page
    st.write('''This application enables users to predict the sentiment of hotel reviews. 
            By entering a review, users can gain insights into its sentiment (Negative, Neutral, or Positive).''')

    # Get user input
    user_input = st.text_area("Enter the review text for sentiment analysis:") 

    if st.button('Analyze'):
        if user_input:
            # Check sentiment using the function
            sentiment = check_sentiment(user_input) 

            # Display the result
            st.write(f"Sentiment: {sentiment}")
        else:
            st.write("Please enter some text.")

# Run the app
if __name__ == "__main__":
    run()