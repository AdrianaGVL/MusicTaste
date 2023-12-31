##############################
#    Author: Adriana Galán
#    Music Taste Project
############################

# Libraries
import pandas as pd
import numpy as np
import os
import librosa
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import features_extration as fx

# Data paths
parent_path = str(Path(Path.cwd()).parents[6])
data_path = main_path = f'{parent_path}/Estudios/Universidad/Máster/PRDL+MLLB/used_dataset'
genres = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]
freq_features = [f"MFCC {num}" for num in range(1, 21)]
others_features = ['Beats_song', 'Danceability', 'Loudness', 'Spectral_Rolloff', 'Spectral Centroid', 'Energy']
features = others_features + freq_features # + ['Mark']
# genre_preferences = {'Alternative': 0.7, 'Pop': 0.8, 'Techno': 0.4, 'Rock': 0.3, 'Dance': 0.6, 'Classical': 0.2}

# DataFrames creation
# regresdf = pd.DataFrame(columns=["Genre"] + regression_features)
# classidf = pd.DataFrame(columns=["Genre"] + classification_features)
df = pd.DataFrame(columns=["Genre"] + features)

# Add data to the  DataFrames
for genre in genres:
    # Music path
    music = f'{data_path}/{genre}'
    songs = os.listdir(music)
    print(f'Number of files in {genre} folder: {len(songs)}')
    for song in songs:
        file, extension = os.path.splitext(song)
        if not extension == '.mp3':
            print(f'The {genre} folder not only contains songs')
            print(f'\tNot valid file: {file}')
            songs.remove(file)
            continue

        # Feature extraction
        audio, sr = librosa.load(f'{music}/{song}')

        # MFCC
        MFCCs = fx.get_mel(audio, sr)

        # Tempo and beats per song
        # Tempo won't be use because its poor correlation to the genre
        beats = fx.get_tempo(audio, sr)
        total_beats = len(beats)

        # Loudness
        loudness = fx.get_loudness(audio)

        # Danceability
        danceability, _ = fx.get_danceability(audio)

        # Energy
        energy = fx.get_energy(audio)

        # Spectrum
        # Check if the length is odd
        if len(audio) % 2 != 0:
            audio = np.append(audio, 0)

        spectrum = fx.get_spectrum(audio)

        # Spectral Roll-off
        roff = fx.get_spect_roff(spectrum)

        # Spectral Centroid
        centroid = fx.get_spect_centroid(spectrum)

        # Regression dataframe
        new_row_r = MFCCs.tolist()
        # regresdf.loc[len(regresdf.index), :] = new_row_r
        # Classification dataframe
        new_row_c = [genre, total_beats, danceability, loudness, roff, centroid, energy]
        # classidf.loc[len(classidf.index), :] = new_row_c

        # Whole dataframe
        # row_to_add = []
        # row_to_add.append(new_row_c)
        # row_to_add.append(new_row_c)
        df.loc[len(df.index), :] = new_row_c + new_row_r

# Dataframe - With Energy
df_path = f'{str(Path(Path.cwd()))}/new_data'
df.to_csv(f'{df_path}/df_energy.csv', sep=';', decimal=",", index=False)

# Normalise each column with Z-score (mean = 0, std = 1)
standard_scaler = StandardScaler()
# General dataset
for feat in features:
    df[feat] = standard_scaler.fit_transform(df[[feat]])

# Normalised dataframe
df_path = f'{str(Path(Path.cwd()))}/new_data'
df.to_csv(f'{df_path}/df_enorm.csv', sep=';', decimal=",", index=False)