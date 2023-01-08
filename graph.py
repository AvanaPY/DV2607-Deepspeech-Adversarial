"""
    DV2607 - Project
    Written by
        Emil Karlström
        Samuel Jonsson
"""
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
    mean = np.mean(waveform)
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    mean_axis = torch.ones(size=(num_frames,)) * mean

    figure = plt.gcf()
    axes = plt.subplot(subplot)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].plot(time_axis, mean_axis, linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        axes[c].set_title(title)

org, _ = torchaudio.load('audio16k/osr34.wav')
audio, sample_rate = torchaudio.load('save.wav')
print(f'Original and attacked data:')
print(org)
print(audio)

print(f'Mean of original: {torch.mean(org)}')

transform = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
db_ori = transform(org)
db_new = transform(audio)
estimated_noise = audio - org
db_noise = 20* torch.max(transform(estimated_noise)/200)
# abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(org.numpy())**2)))
# abs_new = 20*np.log10(np.sqrt(np.mean(np.absolute(audio.numpy())**2)))
db_diff = 20 * torch.max((db_new - db_ori)/200)
print(f'dB data:')
print(db_ori)
print(db_new)
print(f'Difference: {db_diff}')
print(f'Noise dB: {db_noise}')
plot_waveform(org, sample_rate, 222, "Original")
plot_waveform(audio, sample_rate, 224, "Attacked")

plot_specgram(org, sample_rate, 221, title="Original")
plot_specgram(audio, sample_rate, 223, "Attacked")
plt.show()