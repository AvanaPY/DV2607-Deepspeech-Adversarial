import os
import sys
sys.path.append("deepspeech.pytorch/")
import torchaudio
import time
import argparse
from model import Attacker
from model import load_model_and_decoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-file', type=str, required=True, dest='audio_file', help='Path to audio file')
    parser.add_argument('--target-sentence', type=str, default=None, dest='target_sentence', help='Which target sentence to aim for')
    parser.add_argument('--attack-method', choices=['pgd', 'fgsm', 'untargeted'], default='fgsm', help='Which adversarial attack method to use')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Which value of epsilon to use for PGD and FGSM attacks')
    parser.add_argument('--alpha', type=float, default=0.01, help='Which value of alpha to use for PGD attack')
    parser.add_argument('--pgd-steps', type=int, default=50, help='Number of PGD iterations', dest='pgd_steps')
    parser.add_argument('--force-download-model', type=bool, default=False, 
                            dest='force_download_model', help='Whether or not to force a redownload of the model')
    args = parser.parse_args()
    
    DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    model_path_rel      = 'models/librispeech/librispeech_pretrained_v3.ckpt'
    model_download_url  = 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt'
    
    model_path_abs = os.path.join(DIRECTORY, model_path_rel)
    model, decoder = load_model_and_decoder(model_path_abs, model_download_url, args.force_download_model)

    audio_path = os.path.join(DIRECTORY, args.audio_file)
    sound, sample_rate = torchaudio.load(audio_path)
    
    if args.target_sentence:
        target_str = args.target_sentence.upper()
    else:
        target_str = None
        
    attacker = Attacker(model=model, 
                        decoder=decoder, 
                        sound=sound, 
                        target_str=target_str,
                        attack_method=args.attack_method)
    original, final, db_diff, l_distance = attacker.attack(
        epsilon=args.epsilon,
        alpha=args.alpha,
        pgd_rounds=args.pgd_steps
    )
    
    print(f'Original and final outputs:')
    print(original)
    print()
    print(final)
    print()
    
    print(f'dB difference: {db_diff}')
    print(f'Levenshtein distance: {l_distance}')