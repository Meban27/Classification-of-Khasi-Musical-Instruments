import streamlit as st
import numpy as np
import librosa
import joblib
import pandas as pd
from PIL import Image

# Load your trained model
model = joblib.load('khasi.pkl')


# Feature extraction function
def extract_features(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    frame_size = 1040  # Adjust as needed
    hop_length = 504   # Adjust as needed
    
    # Extract features
    amplitude_envelope = np.abs(librosa.util.frame(y, frame_length=frame_size,
                                                   hop_length=hop_length)).max(axis=0)
        # Smooth the envelope using a rolling mean
    smoothed_envelope= np.convolve(amplitude_envelope, 
                            np.ones(10)/10, mode='same')
    adsr=np.mean(smoothed_envelope)
    
    zcr=librosa.feature.zero_crossing_rate(y)[0]
    zero_crossing_rate=np.mean(zcr)
    flux= librosa.onset.onset_strength(y=y, sr=sr)
    cent=librosa.feature.spectral_centroid(y=y,sr=sr)[0]
    spectral_centroid=np.mean(cent)
    roll=librosa.feature.spectral_rolloff(y=y,sr=sr)[0]
    spectral_rolloff= np.mean(roll)
    flux= librosa.onset.onset_strength(y=y, sr=sr)
    spectral_flux=np.mean(flux)
    # Create a DataFrame
    features = pd.DataFrame([[adsr, zero_crossing_rate, spectral_centroid, spectral_rolloff, spectral_flux]],
                            columns=['adsr', 'zero_crossing_rate', 'spectral_centroid', 'spectral_rolloff', 'spectral_flux'])
    return features


logo = Image.open('logo.jpg')  # Ensure 'logo.png' is in the same directory
desired_width = 150  # Adjust as needed
aspect_ratio = logo.size[1] / logo.size[0]
new_size = (desired_width, int(desired_width * aspect_ratio))
logo_resized = logo.resize(new_size)

st.sidebar.image(logo_resized, use_column_width=False)

page = st.sidebar.radio("# Navigation", ["Home", "About"])

if page == "Home":
# Create the Streamlit app
    st.title("Khasi Instrument Prediction from Audio")

    # Upload audio file
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        # Extract features
        features = extract_features(uploaded_file)
        st.audio(uploaded_file)
        
        # Predict the instrument
        prediction = model.predict(features)

        if(prediction==1):
            st.write("## The instrument is a Duitara")
            st.image("duitara.jpg",caption='Duitara')
        elif(prediction==2):
            st.write("## The instrument is an Singphong")
            st.image("singphong.jpg",caption='Singphong')
        elif(prediction==3):
            st.write("## The instrument is a Besli ")
            st.image("besli.jpg",caption='Besli')
        elif(prediction==4):
            st.write("## The instrument is a Bom")
            st.image("Bom.jpg",caption='Bom')
        elif(prediction==5):
            st.write("## The instrument is a Ksing Kynthei")
            st.image("ksing kynthei.jpg",caption='Ksing kynthei')
        elif(prediction==6):
            st.write("## The instrument is a Ksing Shyngrang")
            st.image("ksing shynrang.jpg",caption='Ksing Shynrang')
        elif(prediction==7):
            st.write("## The instrument is a Pdiah")
            st.image("pdiah.jpg",caption='Pdiah')
        

        
        # Display the result
        

        # Optionally display extracted features
        st.write("Extracted Features:")
        st.write(features)
elif page == "About":
    # About page content
    
    st.write("""
# Classification of Khasi Musical Instruments
This web application serves as an educational platform, showcasing the final semester project of Group #2 from the NEHU B.Tech 2020-24 batch. The project focuses on the classification of Khasi musical instruments, developed by:

- Mebanpynshai Lyngdoh Marshilong (21BtechLECE07)
- Moirangthem Bhorot Singh (20BTechECE)
- Yongyo HM (21BtechLECE01)

## Project Overview
Our primary goal is to bridge the gap between Khasi musical instruments and modern technology. This project aims to analyze and classify these traditional instruments, thereby promoting their rich cultural heritage and encouraging further academic and technological exploration.

## Features
Audio File Upload: Users can upload an audio file of a Khasi musical instrument. The app will analyze the audio and predict the instrument type.
Instrument Prediction: Leveraging advanced machine learning techniques, the app identifies the instrument from the uploaded audio.
Informative Results: After prediction, the app provides information about the identified instrument, accompanied by relevant images.

## Datasets
Due to an agreement with the Department of Culture and Creative, NEHU Shillong, the datasets used in this project are not publicly available. We invite students and researchers to contribute by collecting additional datasets and proposing innovative enhancements to further this project.

## Contributions
This project is developed for educational and non-profit purposes. It should not be used for personal gain but rather to foster further development in this field. We welcome contributions such as new data, improved algorithms, or other enhancements. If you have ideas or improvements, feel free to fork the repository and submit a pull request.

## Future Work
This project lays the groundwork for the integration of Khasi musical instruments with modern technology. We hope it will inspire continued research and development. Potential future work includes:

- Expanding the dataset with more diverse and comprehensive samples.
- Enhancing the classification algorithms for better accuracy.
- Developing interactive applications or interfaces for Khasi musical instruments.
- We are excited to see how this project evolves and improves over time, contributing to the preservation and celebration of Khasi musical heritage.
""")
