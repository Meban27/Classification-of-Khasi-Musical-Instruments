import streamlit as st
import numpy as np
import librosa
import joblib
import pandas as pd

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

# Create the Streamlit app
st.title("Instrument Prediction from Audio")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Extract features
    features = extract_features(uploaded_file)
    
    # Predict the instrument
    prediction = model.predict(features)

    if(prediction==1):
        st.write("The instrument is a Duitara")
    elif(prediction==2):
        st.write("The instrument is an Singphong")
    elif(prediction==3):
        st.write("The instrument is a Besli ")
    elif(prediction==4):
        st.write("The instrument is a Bom")
    elif(prediction==5):
        st.write("The instrument is a Ksing Kynthei")
    elif(prediction==6):
        st.write("The instrument is a Ksing Shyngrang")
    elif(prediction==7):
        st.write("The instrument is a Pdiah")
    

    
    # Display the result
    #st.write(f"Predicted Instrument: {prediction[0]}")

    # Optionally display extracted features
    st.write("Extracted Features:")
    st.write(features)

