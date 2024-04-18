import streamlit as st
import pandas as pd
import keras
import pickle
import cv2
import numpy as np

# Set page title and favicon
st.set_page_config(page_title="Road Sign Classification", page_icon="🚦")

# Add title and image header
st.title("Road Sign Classification")
st.image("/Users/amitprajapati/Documents/WPI_DOCS/Courses/ML/archive/Roadsignimages.png", use_column_width='always')

# File uploader for image
data = st.file_uploader("Upload image")

# Function to classify image
def classify_image(img):
    img = cv2.resize(img, (100, 100))
    img = img.reshape(1, 100, 100, 3) / 255
    reconstructed_model = keras.models.load_model("/Users/amitprajapati/Documents/WPI_DOCS/Courses/ML/archive/CNN_OVER_UNDER_SAMPLED_DATA.keras")
    Mapping = pd.read_csv('Transformation.csv')
    pred = np.argmax(reconstructed_model.predict(img), axis=1)
    predicted_class = Mapping.loc[Mapping['values'] == pred[0], 'keys'].iloc[0]
    return predicted_class.upper()

# Display image and prediction
if data is not None:
    file_bytes = np.asarray(bytearray(data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predicted_class = classify_image(img)
    
    # Display prediction
    st.write(f'Predicted Class: {predicted_class}')
    
    # Display resized image
    st.image(img, caption='Uploaded Image', use_column_width=True)
