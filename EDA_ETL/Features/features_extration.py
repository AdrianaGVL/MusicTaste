##############################
#   Author: Adriana Galán
#   Music Taste Project
############################

# Libraries
import librosa
import numpy as np
import essentia.standard as ess
import statistics
from EDA_ETL.Features import charts as ch


# Mel-spectrum
def get_mel(track, sr):
    mfccs = librosa.feature.mfcc(y=track, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Covariance and Correlation Normalised Matrices
    covariance_matrix = np.cov(mfccs)
    covariance_matrix_normalized = np.nan_to_num(
        covariance_matrix / np.sqrt(np.outer(np.diag(covariance_matrix), np.diag(covariance_matrix))))
    correlation_matrix = np.corrcoef(mfccs)
    correlation_matrix_normalized = np.nan_to_num(
        correlation_matrix / np.sqrt(np.outer(np.diag(correlation_matrix), np.diag(correlation_matrix))))

    # # Feature Study
    # # Because the huge amount of data each MFCC coefficient contains
    # # a study of how ir correlated is done to see if it's possible
    # # to reduce it to some values.
    # ch.cov_corr(mfccs.shape[0], correlation_matrix_normalized, 'Correlation', safe=True)
    # ch.cov_corr(mfccs.shape[0], covariance_matrix_normalized, 'Covariance', safe=True)

    # # Selection of desired MFCC coefficients
    # # These are the ones selected because first and last coefficient
    # # are the most significantly different. MFCC 10 coefficient
    # # value is selected because is just in the middle.
    # mfcc1 = mfccs[0, :]
    # mfcc10 = mfccs[9, :]
    # mfcc20 = mfccs[19, :]

    # Mean, max. and min
    # mfcc1_mean = np.mean(mfcc1, axis=0)
    # mfcc1_min = np.min(mfcc1, axis=0)
    # mfcc1_max = np.max(mfcc1, axis=0)

    # mfcc10_mean = np.mean(mfcc1, axis=0)
    # mfcc10_min = np.min(mfcc1, axis=0)
    # mfcc10_max = np.max(mfcc1, axis=0)

    # mfcc20_mean = np.mean(mfcc1, axis=0)
    # mfcc20_min = np.min(mfcc1, axis=0)
    # mfcc20_max = np.max(mfcc1, axis=0)

    # Just the first value of each MFCC
    # mfcc_first = mfccs[:, 0]

    return mfccs_mean


# Spectrogram
def get_ft(track):
    fourier_transform = librosa.stft(track)
    spectro = np.abs(fourier_transform)

    # Feature Study
    ch.spectrogram(spectro)
    return spectro


# Chroma
def get_chroma(track, sr):
    chroma = librosa.feature.chroma_stft(y=track, sr=sr)

    # Feature Study
    # ch.chroma_song(chroma)

    return chroma


# Tempo
def get_tempo(track, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=track, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return beat_times


def get_loudness(track):
    loudness = ess.Loudness()(track)
    loudness = 10 * np.log10(loudness)
    return loudness


def get_danceability(track):
    danceability = ess.Danceability()(track)
    return danceability



def get_energy(track):
    energy = ess.Energy()(track)
    energy = 10 * np.log10(energy)
    return energy


def get_spectrum(track):
    spectrum = ess.Spectrum()(track)
    return spectrum


def get_spect_roff(spectrum):
    rolloff = ess.RollOff()(spectrum)
    return rolloff


def get_spect_centroid(spectrum):
    centroid = ess.Centroid()(spectrum)
    return centroid


def marks(leng, percentage):
    # Depending on the genre a percentage of the marks will be great ones
    high_marks = np.random.normal(loc=8.5, scale=1.5, size=int(leng * percentage))
    high_marks = np.clip(high_marks, 7, 10)
    # The rest will  be bas ones
    low_marks = np.random.normal(loc=3, scale=1.5, size=int(leng * (1-percentage)))
    low_marks = np.clip(low_marks, 0, 6)

    marks = np.concatenate([high_marks, low_marks])
    np.random.shuffle(marks)

    return marks