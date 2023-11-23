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
# regression_features = [f"MFCC {num} mean" for num in range(10, 19)] + [f"MFCC {num} min" for num in range(2, 19)] + [f"MFCC {num} max" for num in range(2, 19)]
regression_features = ['MFCC 1 mean', 'MFCC 1 min',' MFCC 1 max', 'MFCC 10 mean', 'MFCC 10 min', 'MFCC 10 max', 'MFCC 20 mean', 'MFCC 20 min', 'MFCC 20 max']
classification_features = ['Tempo', 'Beats per song', 'Danceability', 'Loudness (dB)', 'Energy (dB)', 'Spectral Rolloff', 'Spectral Centroid']
features = classification_features + regression_features

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

        # Tempo
        tempo, beats = fx.get_tempo(audio, sr)
        total_beats = len(beats)

        # MFCC
        # meanMFCC, minMFCC, maxMFCC = fx.get_mel(audio, sr)
        # new_row = [genre] + meanMFCC.tolist() + minMFCC.tolist() + maxMFCC.tolist()
        MFCCs = fx.get_mel(audio, sr)
        # new_row_r = [genre] + MFCCs
        # regresdf.loc[len(regresdf.index), :] = new_row_r

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

        # Classification dataframe
        new_row_c = [genre, tempo, total_beats, danceability, loudness, energy, roff, centroid]
        # classidf.loc[len(classidf.index), :] = new_row_c

        # Whole dataframe
        df.loc[len(df.index), :] = new_row_c + MFCCs

# General dataframe
df_path = f'{str(Path(Path.cwd()))}/new_data'
df.to_csv(f'{df_path}/df.csv', sep=';', decimal=",", index=False)

# Assing a number to each genre
# regresdf.Genre = regresdf.Genre.map({'Alternative': 0, 'Pop': 1, 'Techno': 2, 'Dance': 3, 'Rock': 4, 'Classical': 5})
# classidf.Genre = classidf.Genre.map({'Alternative': 0, 'Pop': 1, 'Techno': 2, 'Dance': 3, 'Rock': 4, 'Classical': 5})

# # Normalise each column with Z-score (mean =, std = 1)
# standard_scaler = StandardScaler()
# #Regression
# cols_to_norm = regression_features
# regresdf[cols_to_norm] = standard_scaler.fit_transform(regresdf[cols_to_norm])
# # Classification
# cols_to_norm = classification_features
# classidf[cols_to_norm] = standard_scaler.fit_transform(classidf[cols_to_norm])
#
#  # Save both dataframes
# regressdf_path = f'{str(Path(Path.cwd()))}/new_data'
# classidf_path = f'{str(Path(Path.cwd()))}/new_data'
# regresdf.to_csv(f'{regressdf_path}/regressdf.csv, sep='\t', decimal=",", index=False)
# classidf.to_csv(f'{classidf_path}/classidf.csv, sep='\t', decimal=",", index=Falsee)