import streamlit as st
import pandas as pd
import keras
import pickle
from PIL import Image
import numpy as np

# Set page title and favicon
st.set_page_config(page_title="Road Sign Classification", page_icon="🚦")

# Custom styles
st.markdown("""
<style>
.main { 
    background-color: #f0f2f6;
}
h1 {
    color: #333366;
}
.streamlit-expanderHeader {
    font-size: 16px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Add title and image header
st.title("Road Sign Classification")
st.image("/Users/amitprajapati/Documents/WPI_DOCS/Courses/ML/archive/Roadsignimages.png", use_column_width='always')

# File uploader for image
data = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])

# Function to classify image
def classify_image(img):
    img = img.resize((100, 100))  # Resize image
    img = np.array(img)  # Convert image to numpy array
    img = img.reshape(1, 100, 100, 3) / 255  # Reshape and normalize
    reconstructed_model = keras.models.load_model("/Users/amitprajapati/Documents/WPI_DOCS/Courses/ML/archive/CNN_OVER_UNDER_SAMPLED_DATA.keras")
    Mapping = pd.read_csv('Transformation.csv')
    pred = np.argmax(reconstructed_model.predict(img), axis=1)
    predicted_class = Mapping.loc[Mapping['values'] == pred[0], 'keys'].iloc[0]
    return predicted_class.upper()

# Display image and prediction
if data is not None:
    img = Image.open(data)  # Open image
    predicted_class = classify_image(img)
    
    # Display prediction
    st.success(f'Predicted Class: {predicted_class}')
    
    # Display resized image
    st.image(img, caption='Uploaded Image', use_column_width=True)

# Add some interactive components
with st.expander("ℹ️ - About this app", expanded=True):
    st.write("""
    This application is designed to classify traffic signs in real time using a deep learning model. Simply upload an image of a traffic sign, and the system will predict its class.
    """)

# Footer
st.markdown("---")
st.markdown("🚀 App built with Streamlit | 🛠️ by [Your Name]")
