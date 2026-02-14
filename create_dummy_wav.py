import numpy as np
import soundfile as sf

# Generate 1 second of silence/noise
sr = 16000
audio = np.random.uniform(-0.1, 0.1, sr)
sf.write('dummy_audio.wav', audio, sr)
print("Created dummy_audio.wav")
