# Import necessary libraries
import streamlit as st
import nltk
import pickle
import string
from nltk.stem import PorterStemmer

Porter = PorterStemmer()

# Download necessary NLTK resources
nltk.download('stopwords')  # Download stopwords
nltk.download('punkt')      # Download punkt tokenizer
nltk.download('wordnet')    # Download wordnet for stemming
from nltk.corpus import stopwords

stopwords.words('english')  # Access stopwords for English language

def final_transformed_text(text):
    """
    Function to preprocess text for classification.
    
    Args:
    - text (str): Input text to be processed.
    
    Returns:
    - str: Processed and transformed text.
    """
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text into words

    # Filter out non-alphanumeric tokens
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    filtered = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stem words using Porter stemmer
    filtered_final = [Porter.stem(i) for i in filtered]

    return " ".join(filtered_final)


# Load pre-trained TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))  # Load TF-IDF vectorizer from file
model = pickle.load(open('model.pkl', 'rb'))      # Load classification model from file

# Streamlit UI components
st.title('Email Spam/Ham Classification')  # Title of the web application
input_text = st.text_input('Please enter your email address')  # Input text box for user input

if st.button('Check'):  # Button to trigger classification
    final_text = final_transformed_text(input_text)  # Preprocess input text

    tfidf_vector = tfidf.transform([final_text])  # Transform preprocessed text using TF-IDF vectorizer

    tfidf_vector = tfidf_vector.toarray()  # Convert TF-IDF vector to array format for model input

    result = model.predict(tfidf_vector)[0]  # Predict using loaded model

    # Display result based on prediction
    if result == 1:
        st.header('Spam')  # Display header indicating email is predicted as spam
    else:
        st.header('Ham')   # Display header indicating email is predicted as ham (not spam)
