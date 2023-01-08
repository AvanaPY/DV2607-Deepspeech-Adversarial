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
import time
import argparse
import torch

from model.model import Attacker
from model.model import load_model_and_decoder

alpha   = 0.00001

target_sentence = 'Martin told us that our project was the best performed one in the class'.upper()

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
model_path_rel = 'models/librispeech/librispeech_pretrained_v3.ckpt'

for audio_file_name in ['osr34']:
    print(f'Doing Audio {audio_file_name}.wav')
    for attack_method in ['untargeted']:
        print(f'\tDoing method {attack_method}')
        
        step_list = [400, 700, 1000]
        for steps in step_list:
            print(f'\t\tFor steps {steps}')
            # for epsilon in [0.2, 0.1, 0.01, 0.001, 0.0001]:
            
            eps_list = [0.001]
                
            for epsilon in eps_list:
                print(f'\t\t  Epsilon {epsilon}')

                model_path_abs = os.path.join(DIRECTORY, model_path_rel)
                model, decoder = load_model_and_decoder(model_path_abs, None, False)

                audio_file      = f'audio16k/{audio_file_name}.wav'
                audio_path = os.path.join(DIRECTORY, audio_file)
                sound, sample_rate = torchaudio.load(audio_path)
                
                attacker = Attacker(model=model, 
                                    decoder=decoder, 
                                    sound=sound, 
                                    target_str=target_sentence,
                                    attack_method=attack_method,
                                    save_path=os.path.join(DIRECTORY, f'saved_wav/{audio_file_name}_{attack_method}_{steps:3d}_{epsilon:3f}_{alpha:5f}.wav'))
                original, final, db_diff, l_distance, (original_audio, final_audio) = attacker.attack(
                    epsilon=epsilon,
                    alpha=alpha,
                    pgd_rounds=steps,
                    return_audio=True
                )

                transform = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
                db_ori = transform(original_audio)
                db_new = transform(final_audio)
                estimated_noise = final_audio - original_audio
                db_noise = 20* torch.max(transform(estimated_noise)/200)
                db_diff = 20 * torch.max((db_new - db_ori)/200)
                print(f'Final output (L-distance: {l_distance:3d}, noise dB: {db_noise:4.1f})')
                # print(final)
                # print()