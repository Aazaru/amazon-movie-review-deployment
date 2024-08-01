import joblib
import numpy as np
import streamlit as st
import pandas as pd

# Load the saved preprocessors and model
vectorizer = joblib.load(r'C:\Users\aazar\OneDrive\Desktop\Data Science projects\TDIDF_VECTOZIER.pkl')
model = joblib.load(r'C:\Users\aazar\OneDrive\Desktop\Data Science projects\xgboost_model.pkl')
label_encoder = joblib.load(r'C:\Users\aazar\OneDrive\Desktop\Data Science projects\label_encoder.pkl')

# Known categories and prediction mapping
known_categories = ['Amazon Video', 'DVD', 'Blu-ray', 'VHS Tape', 'HD DVD', 'MP3 Music', 'Audio CD']
prediction_map = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}

# Streamlit app
st.title('Sentiment Analysis Prediction')

# Input fields
movie_name = st.text_input('Enter movie name:', '')
text = st.text_input('Enter your reviews:', '')
format = st.radio('Select format:', known_categories)

# Radio buttons for rating
rating = st.radio('Select rating:', [1, 2, 3, 4, 5])

if st.button('Predict'):
    # Preprocess the text feature
    text_features = vectorizer.transform([text])
    text_features_df = pd.DataFrame(text_features.toarray())

    # Encode the format feature
    format_features = label_encoder.transform([format]).reshape(-1, 1)
    format_features_df = pd.DataFrame(format_features, columns=['Format'])

    # Prepare the rating feature
    rating_features_df = pd.DataFrame([[rating]], columns=['Rating'])

    # Combine all features using pandas.concat
    combined_features_df = pd.concat([text_features_df, format_features_df, rating_features_df], axis=1)

    # Convert DataFrame to numpy array for model prediction
    combined_features = combined_features_df.values

    # Make the prediction
    prediction = model.predict(combined_features)

    # Map the prediction to labels
    prediction_label = prediction_map.get(prediction[0], 'Unknown')

    # Display results
    st.write(f'Prediction: {prediction_label}')
    st.write(f'Movie Name: {movie_name}')