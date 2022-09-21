import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime

num = 60

for x in range(num):
    fs = 44100
    duration = 5  # seconds
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")


    myrecording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype='float64')
    print("Recording Audio " + str(x + 1) + "/" + str(num))

    # Start recorder with the given values
    # of duration and sample frequency
    sd.wait()

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write(f'recording_{date}.wav', fs, myrecording)
    print("Audio recording complete")

    # Convert the NumPy array to audio file
    # wv.write("recording1.wav", myrecording, fs, sampwidth=2)
    # resource: https://www.geeksforgeeks.org/create-a-voice-recorder-using-python/
