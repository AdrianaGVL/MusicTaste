##############################
#    Author: Adriana Galán
#   Music Taste Project
############################

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import librosa
from pathlib import Path


def cov_corr(num_mfcc, matrix, cov_or_corr, safe=False, somethingelse=""):
    # Labels
    mfcc_labels = [f'MFCC {i + 1}' for i in range(num_mfcc)]

    # Covariance Charts
    plt.figure(figsize=(20, 18))
    plt.imshow(matrix, cmap='coolwarm_r', origin='lower', aspect='auto')
    plt.xticks(np.arange(num_mfcc), mfcc_labels, rotation=45)
    plt.yticks(np.arange(num_mfcc), mfcc_labels)
    plt.title(f'{cov_or_corr} Matrix between MFCCs vales')
    plt.colorbar(label=f'{cov_or_corr} Value')

    if safe:
        plt.savefig(f'{str(Path(Path.cwd()))}/new_data/{cov_or_corr}_matrix{somethingelse}.png')

    plt.show()

    return


def chroma_song(chroma):
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.title('Chromagram')
    plt.colorbar()
    plt.show()

    return


def spectrogram(spectro):
    librosa.display.specshow(librosa.amplitude_to_db(spectro, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Lineal Spectrogram')
    plt.show()

    return
