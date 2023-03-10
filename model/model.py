"""
    DV2607 - Project
    Written by
        Emil Karlström
        Samuel Jonsson
"""
import torch
import torch.nn as nn
import numpy as np
import Levenshtein
import torchaudio

from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.configs.inference_config import TranscribeConfig

from model.stft import STFT
from model.model_downloader import verify_model_exist

SAMPLE_RATE = 16_000

def target_str_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for word in sentence:
        out.append(labels.index(word))
    return torch.IntTensor(out)

def spectrogram(sound, stft):
    # real, imag = stft(sound)
    # mag, cos, sin = magphase(real, imag)
    mag, _ = stft(sound)
    mag = torch.log1p(mag)
    mean = mag.mean()
    std = mag.std()
    mag = mag - mean
    mag = mag / std
    mag = mag.permute(0,1,3,2)
    return mag

def load_model_and_decoder(model_path_abs : str, 
                           model_download_url : str, 
                           force_download_model : bool = False):
    
    cfg = TranscribeConfig
    verify_model_exist(model_path_abs, 
                       model_download_url, 
                       force_download_model)

    try:
        model = load_model(device="cpu", model_path=model_path_abs)
        decoder = load_decoder(labels=model.labels, cfg=cfg.lm)
        return model, decoder
    except Exception as e:
        print(f'Error loading model {model_path_abs}, file may be corrupted, you may force-download it again with the --force-download-model option set to True')
        exit(0)

class Attacker:
    def __init__(self, model, 
                 decoder, 
                 sound, 
                 target_str, 
                 attack_method : str = 'fgsm',
                 save_path : str = 'save.wav',
                 device : str = 'cuda'):
        self._device = device
        self._attack_method = attack_method
        self._model = model
        self._model.to(self._device)
        self._model.train()              # Set the model into train mode so we can use backward() on the RNN
        self._decoder = decoder
        self._loss = nn.CTCLoss()
        self._sound = sound
        self._save_path = save_path
        
        if self._attack_method != 'untargeted':
            assert not (target_str is None), "You must have a target string for targeted attacks"
            self._target_string = target_str
            self._target = target_str_to_label(target_str)
            self._target = self._target.view(1,-1)
            self._target_length = torch.IntTensor([self._target.shape[1]]).view(1,-1)
        else:
            self._target_string = None
            self._target = None
            self._target_length = None
        
        n_fft       = int(SAMPLE_RATE * 0.02)
        hop_length  = int(SAMPLE_RATE * 0.01)
        win_length  = int(SAMPLE_RATE * 0.01)
        self.stft = STFT(   n_fft=n_fft, 
                            hop_length=hop_length, 
                            win_length=win_length,  
                            window='hann',
                            device=self._device)
        
    @property
    def device(self):
        return self._device
    
    @property
    def model(self):
        return self._model
    
    @property
    def decoder(self):
        return self._decoder
    
    def attack(self, epsilon : float = 0.1, alpha : float = 1e-2, pgd_rounds : int = 50, return_audio : bool = False):
        
        data = self._sound.to(self.device)
        data_raw = data.clone().detach()
        
        spec = spectrogram(data, self.stft)
        input_size = torch.IntTensor([spec.size(3)]).int()
        out, output_size, _ = self._model(spec, input_size)
        decoded_output, decoded_offsets = self._decoder.decode(out, output_size)
        original_output = decoded_output[0][0]
       
        target = self._target
        if target is None:
            target_str = original_output
            target = target_str_to_label(target_str).view(1,-1)
            target_length = torch.IntTensor([target.shape[1]]).view(1,-1)
        else:
            target_str = self._target_string
            target = target.to(self.device)
            target_length = self._target_length
        
        # Let us perform some attack
        if self._attack_method == 'fgsm':
            perturbed_data = self.fgsm_attack(epsilon, data, target, target_length)
            
        elif self._attack_method == 'pgd':
            perturbed_data = self.pgd_attack(epsilon, alpha, pgd_rounds, data, data_raw, target, target_length)     

        elif self._attack_method == 'untargeted':  
            perturbed_data = self.untargeted_attack(epsilon, alpha, pgd_rounds, data, data_raw, target, target_length)     
        
        else:
            raise NotImplementedError(f'Attack method {self._attack_method} not implemented.')
        spec = spectrogram(perturbed_data, self.stft)
        input_size = torch.IntTensor([spec.size(3)]).int()
        out, output_size, _ = self.model(spec, input_size)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_size)
        final_output = decoded_output[0][0]
        
        db_ori  = 20 * np.max(np.log10(np.absolute(data_raw.cpu().detach().numpy())))
        db_aft  = 20 * np.max(np.log10(np.absolute(perturbed_data.cpu().detach().numpy())))
        db_difference = db_aft - db_ori
        levenshtein_distance = Levenshtein.distance(target_str, final_output)
        
        if self._save_path:
            print(f'Saving to {self._save_path}')
            torchaudio.save(self._save_path, src=perturbed_data.cpu(), sample_rate=SAMPLE_RATE)
        
        if not return_audio:
            return original_output, final_output, db_difference, levenshtein_distance
        else:
            return original_output, final_output, db_difference, levenshtein_distance, (data_raw, perturbed_data)
    
    # This implements a version of an untargeted white box attack
    # which is based on the same algorithm of PGD, however instead of minimisng
    # the loss to a target sentence, it tries to maximise the loss against
    # the original data.
    def untargeted_attack(self, epsilon, alpha, pgd_rounds, data, data_raw, target, target_length):
        for round in range(pgd_rounds):
            done = int(50 * round / pgd_rounds)
            print(f'\rUntargeted [{"="*done}{" "*(50-done)}] {round:6,d} / {pgd_rounds:6,d}', end='')
            data.requires_grad = True
            
            spec = spectrogram(data, self.stft)
            input_size = torch.IntTensor([spec.size(3)]).int()
            out, output_size, _ = self._model(spec, input_size)
            out = out.transpose(0, 1).log_softmax(2)
            loss = self._loss(out, target, output_size, target_length)
                
            self._model.zero_grad()
            loss.backward()
                
            adversarial_sound = data + alpha * data.grad.data.sign()
            perturbation = torch.clamp(adversarial_sound - data_raw.data, min=-epsilon, max=epsilon)
            data = (data_raw + perturbation).detach_()
            done = int(50 * (round+1) / pgd_rounds)
            print(f'\rPGD [{"="*done}{" "*(50-done)}] {round+1:6,d} / {pgd_rounds:6,d}', end='')
            
        print() 
        perturbed_data = data
        return perturbed_data

    # This implements a basic Projected Gradient Descent attack
    def pgd_attack(self, epsilon, alpha, pgd_rounds, data, data_raw, target, target_length):
        for round in range(pgd_rounds):
            done = int(50 * round / pgd_rounds)
            print(f'\rPGD [{"="*done}{" "*(50-done)}] {round:6,d} / {pgd_rounds:6,d}', end='')
            data.requires_grad = True
            
            spec = spectrogram(data, self.stft)
            input_size = torch.IntTensor([spec.size(3)]).int()
            out, output_size, _ = self._model(spec, input_size)
            out = out.transpose(0, 1).log_softmax(2)
            loss = self._loss(out, target, output_size, target_length)

            self._model.zero_grad()
            loss.backward()
                
            adversarial_sound = data - alpha * data.grad.data.sign()
            perturbation = torch.clamp(adversarial_sound - data_raw.data, min=-epsilon, max=epsilon)
            data = (data_raw + perturbation).detach_()
            done = int(50 * (round+1) / pgd_rounds)
            print(f'\rPGD [{"="*done}{" "*(50-done)}] {round+1:6,d} / {pgd_rounds:6,d}', end='')
            
        print()
            
        perturbed_data = data     
        return perturbed_data

    # This implements a basic FGSM attack
    def fgsm_attack(self, epsilon, data, target, target_length):
        data.requires_grad = True
            
        spec = spectrogram(data, self.stft)
        input_size = torch.IntTensor([spec.size(3)]).int()
        out, output_size, _ = self._model(spec, input_size)
        out = out.transpose(0, 1).log_softmax(2)
        loss = self._loss(out, target, output_size, target_length)
            
        self._model.zero_grad()
        loss.backward()
        
        perturbed_data = data - epsilon * data.grad.data.sign()
        return perturbed_data