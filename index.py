import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
audio_path='hum.wav.wav'
x , sr =librosa.load(audio_path,sr=44100)
print(type(x), type(sr))
ipd.Audio(audio_path)
plt.figure(figsize=(14,5))
librosa.display.waveplot(x,sr=sr)
X=librosa.stft(x)
Xdb=librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
librosa.display.specshow(Xdb,sr=sr,x_axis='time',y_axis='log')
plt.colorbar()
#zooming
n0=9000
n1=9100
plt.figure(figsize=(14,5))
plt.plot(x[n0:n1])
plt.grid()
zero_crossings=librosa.zero_crossings(x[n0:n1],pad=False)
print(sum(zero_crossings))
mfccs=librosa.feature.mfcc(x,sr=sr)
print(mfccs.shape)
librosa.display.specshow(mfccs,sr=sr,x_axis='time')
hop_length=512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
#Autocorrelation
fname='hum.wav.wav'
if fname == 'hum.wav.wav':
    startTime = 0.1                      # of frame, units=seconds
    stopTime = 0.8                       # ditto
elif fname == 'hum.wav.wav':
    startTime = 0.11                     # of frame, units=seconds
    stopTime = 0.18                      # ditto
else:
     assert False
# extract the frame
startIdx = int(startTime * fs)
stopIdx = int(stopTime * fs)
s = signal[startIdx:stopIdx]
phis = []
N = len(s)
for k in range(0, 400):
    phi = 0
    for n in range(k, N):
         phi += s[n] * s[n - k]
    phi = (phi*(1 / N))
    phis.append(phi)
plt.plot(phis)
plt.title('Autocorrelation of Frame')
plt.ylabel('PHI')
plt.xlabel('DELAY INDEX')
plt.autoscale(tight='both')
