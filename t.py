import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

def plot_specgram(waveform, sample_rate, subplot, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure = plt.gcf()
    axes = plt.subplot(subplot)
    # figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        axes[c].set_title(title)
    
def plot_waveform(waveform, sample_rate, subplot, title : str):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure = plt.gcf()
    axes = plt.subplot(subplot)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        axes[c].set_title(title)

org, _ = torchaudio.load('audio16k/osr34.wav')
audio, sample_rate = torchaudio.load('save.wav')
abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(org.numpy())**2)))
abs_new = 20*np.log10(np.sqrt(np.mean(np.absolute(audio.numpy())**2)))
db_diff = abs_new - abs_ori
print(abs_ori)
print(abs_new)
print(f'Difference: {db_diff}')
plot_waveform(org, sample_rate, 222, "Original")
plot_waveform(audio, sample_rate, 224, "Attacked")

plot_specgram(org, sample_rate, 221, title="Original")
plot_specgram(audio, sample_rate, 223, "Attacked")
plt.show()