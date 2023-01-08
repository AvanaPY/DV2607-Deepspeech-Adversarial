"""
    DV2607 - Project
    Written by
        Emil Karlström
        Samuel Jonsson
"""
import os
import sys
sys.path.append("deepspeech.pytorch/")
import torchaudio
import torch
import time
import argparse
import Levenshtein

from model.model import Attacker
from model.model import load_model_and_decoder
from model.model import spectrogram
from model.stft import STFT

martin_text     = "MARTIN TOLD US THAT OUR PROJECT WAS THE BEST PERFORMED ONE IN THE CLASS"
harvard_text    = "THE STALE SMELL OF OLD BEER LINGERS IT TAKES HEAT TO BRING OUT THE ODOR A CALED DIPH RESTORES HEALTH AND ZAST A SALT PICPLED TASTE FINE WITH HAM TACKLOES A'LL PASTDOR ARE MY FAVORITE EZESTFUL FOOD IS THE HOT CROSS BUTTON"
osr34_text      = "A GOLD RING WILL PLEASE MOST ANY GIRL THE LONG JOURNEY HOME TOOK A YEAR SHE SAW A CAT IN THE NEIGHBOUR'S HOUSE A PINK SHELL WAS FOUND ON THE SANDY BEACH SMALL CHILDREN CAME TO SEE HIM THE GRASS AND BUSHES WERE WET WITHA DUWE THE BLIND MAN COUNTED HIS OLD COIINS A SEVERE STORMS TORED DOWN THE BAR SHE CALLED HIS NAME MANY TIMES WHEN YOU HEAR THE BELL COME QUICKLY"
device = torch.device('cuda')

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
model_path_rel      = 'models/librispeech/librispeech_pretrained_v3.ckpt'
model_download_url  = 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt'

model_path_abs = os.path.join(DIRECTORY, model_path_rel)
model, decoder = load_model_and_decoder(model_path_abs, model_download_url, False)
model = model.to(device)
model.train()

sample_rate = 16_000
stft = STFT(
    n_fft=int(sample_rate * 0.02),
    hop_length=int(sample_rate * 0.01),
    win_length=int(sample_rate * 0.01),
    window='hann',
    device=device
)

alpha = 0.00001

for filename in ['harvard', 'osr34']:
    print(f'Audio: {filename}.wav')
    original_file = os.path.join(DIRECTORY, 'audio16k', f'{filename}.wav')
    original_audio, sr = torchaudio.load(original_file)
    
    for method in ['fgsm', 'pgd', 'untargeted']:
        print(f'\tMethod: {method}')
        if method == 'untargeted':
            if filename == 'harvard':
                target_sentence = harvard_text
            elif filename == 'osr34':
                target_sentence = osr34_text
        else:
            target_sentence = martin_text
        
        step_list = [20, 50, 100, 150, 200, 400, 700, 1000]
        if method == 'fgsm':
            step_list = [20]
        for steps in step_list:
            print(f'\t\tSteps: {steps}')
            for eps in [0.2, 0.1, 0.01, 0.001, 0.0001]:

                attacked_filename = f'{filename}_{method}_{steps:3d}_{eps:5f}_{alpha:5f}.wav'
                audio_file = os.path.join(DIRECTORY, 'saved_wav', attacked_filename)
                if not os.path.exists(audio_file):
                    print(f'\t\t  File does not exist')
                    continue
                sound, sample_rate = torchaudio.load(audio_file)
                sound = sound.to(device)
                
                spec = spectrogram(sound, stft)
                input_size = torch.IntTensor([spec.size(3)]).int()
                out, output_size, _ = model(spec, input_size)
                out_decoded, decoded_offsets = decoder.decode(out, output_size)
                out_str = out_decoded[0][0]
                
                d = Levenshtein.distance(out_str, target_sentence)
                    
                original_audio, sr = torchaudio.load(original_file)
                final_audio = sound.cpu().detach_()
                
                transform = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
                db_ori = transform(original_audio)
                db_new = transform(final_audio)
                estimated_noise = final_audio - original_audio
                db_noise = 20* torch.max(transform(estimated_noise)/200)
                db_diff  = 20 * torch.max((db_new - db_ori)/200)
                
                print(f'\t\t  Epsilon {eps:3f}: Levenshtein: {d:3}, Noise: {db_noise:.3f}')
              