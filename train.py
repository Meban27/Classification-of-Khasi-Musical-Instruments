import os
import librosa
import pandas as pd
import numpy as np

def extract_features(file_name):
    # Load audio file
    y, sr = librosa.load(file_name)
    frame_size = 1040  # Adjust as needed
    hop_length = 504   # Adjust as needed
    
    # Extract features
    amplitude_envelope = np.abs(librosa.util.frame(y, frame_length=frame_size,
                                                   hop_length=hop_length)).max(axis=0)
        # Smooth the envelope using a rolling mean
    smoothed_envelope= np.convolve(amplitude_envelope, 
                            np.ones(10)/10, mode='same')
    adsr=np.mean(smoothed_envelope)#extracting ADSR feature
    zero_crossing_rate=librosa.feature.zero_crossing_rate(y)[0]#extracting zero crossing rate feature
    zcr=np.mean(zero_crossing_rate)
    cent=librosa.feature.spectral_centroid(y=y,sr=sr)[0]#extracting spectral centroid feature
    s_c=np.mean(cent)
    roll=librosa.feature.spectral_rolloff(y=y,sr=sr)[0]#extracting spectral rolloff feature
    s_r=np.mean(roll)
    flux= librosa.onset.onset_strength(y=y, sr=sr)#extracting spectral flux rate feature
    s_f=np.mean(flux)

    # Combine features into a single dictionary
    features = {
        'adsr': adsr,
        'zero_crossing_rate': zcr,
        'spectral_centroid': s_c,
        'spectral_rolloff': s_r,
        'spectral_flux':s_f
    }
    return features

def extract_instrument_name(file_name):
    # Extract instrument name from the file name// seperate all the letter/and joining them together
    base_name = os.path.basename(file_name)
    instrument_name = ''.join([i for i in base_name if not i.isdigit()]).split('.')[0]
    return instrument_name

def main():
    # Directory containing the wav files
    wav_dir = 'path/toyour/train/datasets'
    
    # Initialize list to hold feature data
    data = []

    # Process each wav file in the directory
    for file_name in os.listdir(wav_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(wav_dir, file_name)
            features = extract_features(file_path)
            instrument_name = extract_instrument_name(file_name)
            row = {
                'Filename': file_name,
                'Instrument': instrument_name
            }
            # Add features to row dictionary
            row.update(features)
            data.append(row)
    
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)
    instrument_mapping = {
    'duitara': 1,
    'singphong': 2,
    'besli': 3,
    'bom': 4,
    'ksingkynthei': 5,
    'ksingshynrang': 6,
    'pdiah': 7
}

# Apply the mapping to a new Instrument_mapping column and create a new column
    df['Instrument_mapping'] = df['Instrument'].map(instrument_mapping)
    
    # Save the DataFrame to an Excel file,you can use csv or database
    df.to_excel('name/your/excel', index=False)

if __name__ == '__main__':
    main()




