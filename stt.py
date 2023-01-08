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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-file', type=str, required=True, dest='audio_file', help='Path to audio file')
    parser.add_argument('--target-sentence', type=str, required=False, dest='target_sentence', help='A sentence to compare the generated sentence to')
    parser.add_argument('--original-file', type=str, required=False, dest='original_file', help='The original file which can be used to >compute the noise level')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Which device to run the program on')
    args = parser.parse_args()
    
    DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    model_path_rel      = 'models/librispeech/librispeech_pretrained_v3.ckpt'
    model_download_url  = 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt'
    
    model_path_abs = os.path.join(DIRECTORY, model_path_rel)
    model, decoder = load_model_and_decoder(model_path_abs, model_download_url, False)
    
    audio_path = os.path.join(DIRECTORY, args.audio_file)
    sound, sample_rate = torchaudio.load(audio_path)
    device = torch.device(args.device)
    
    sound = sound.to(device)
    model = model.to(device)
    model.train()
    stft = STFT(
        n_fft=int(sample_rate * 0.02),
        hop_length=int(sample_rate * 0.01),
        win_length=int(sample_rate * 0.01),
        window='hann',
        device=device
    )
    
    spec = spectrogram(sound, stft)
    input_size = torch.IntTensor([spec.size(3)]).int()
    out, output_size, _ = model(spec, input_size)
    out_decoded, decoded_offsets = decoder.decode(out, output_size)
    out_str = out_decoded[0][0]
    print()
    print(out_str)
    print()
    
    if args.target_sentence:
        d = Levenshtein.distance(out_str, args.target_sentence.upper())
        print(f'Levenshtein distance to target: {d}')
        
    if args.original_file:
        model.train(False)
        original_audio, sr = torchaudio.load(args.original_file)
        final_audio = sound
        
        transform = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        db_ori = transform(original_audio)
        db_new = transform(final_audio)
        estimated_noise = final_audio - original_audio
        db_noise = 20* torch.max(transform(estimated_noise)/200)
        db_diff = 20 * torch.max((db_new - db_ori)/200)
        print(f'Noise levels in dBs: {db_noise:6.3f}')