import librosa


# Tempo
def get_tempo(track):
    audio = librosa.load(track)
    tempo, beat_frames = librosa.beat.beat_track(y=audio[0], sr=audio[1])
    beat_times = librosa.frames_to_time(beat_frames, sr=audio[1])
    print(f"Estimated Tempo: {tempo:.2f} beats per minute")

    return tempo

