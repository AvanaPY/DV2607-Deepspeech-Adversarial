import torch
import torch.nn as nn
import numpy as np
import Levenshtein
import torchaudio
from stft import STFT, magphase

def target_sentence_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for word in sentence:
        out.append(labels.index(word))
    return torch.IntTensor(out)

def torch_spectrogram(sound, torch_stft):
    real, imag = torch_stft(sound)
    mag, cos, sin = magphase(real, imag)
    mag = torch.log1p(mag)
    mean = mag.mean()
    std = mag.std()
    mag = mag - mean
    mag = mag / std
    mag = mag.permute(0,1,3,2)
    return mag

class Attacker:
    def __init__(self, model, decoder, sound, target_str):
        self._device = 'cuda'
        self._model = model
        self._model.to(self._device)
        self._model.train()              # Set the model into train mode so we can use backward() on the RNN
        self._decoder = decoder
        self._sound = sound
        self._target_string = target_str
        self._target = target_sentence_to_label(target_str)
        self._target = self._target.view(1,-1)
        self._target_lengths = torch.IntTensor([self._target.shape[1]]).view(1,-1)
        
        n_fft       = int(16_000 * 0.02)
        hop_length  = int(16_000 * 0.01)
        win_length  = int(16_000 * 0.01)
        self.stft = STFT(n_fft=n_fft, 
                               hop_length=hop_length, 
                               win_length=win_length,  
                               window='hamming', 
                               center=True, 
                               pad_mode='reflect', 
                               freeze_parameters=True, 
                               device=self.device)
        
    @property
    def device(self):
        return self._device
    
    @property
    def model(self):
        return self._model
    
    @property
    def decoder(self):
        return self._decoder
    
    def attack(self):
        epsilon = 0.1
        alpha   = 1e-3
        pgd_rounds = 20
        
        data, target = self._sound.to(self.device), self._target.to(self.device)
        data_raw = data.clone().detach()
        
        criterion = nn.CTCLoss()
        
        spec = torch_spectrogram(data, self.stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes, _ = self._model(spec, input_sizes)
        decoded_output, decoded_offsets = self._decoder.decode(out, output_sizes)
        original_output = decoded_output[0][0]
        print(f"Original prediction: {decoded_output[0][0]}")
        
        # Let us perform fgsm fuck you
        attack_type = 'pgd'
        if attack_type == 'fgsm':
            data.requires_grad = True
            
            spec = torch_spectrogram(data, self.stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            out, output_sizes, _ = self._model(spec, input_sizes)
            out = out.transpose(0, 1)
            out = out.log_softmax(2)
            loss = criterion(out, target, output_sizes, self._target_lengths)
            
            self._model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            
            # find direction of gradient
            sign_data_grad = data_grad.sign()
            
            # add noise "epilon * direction" to the original sound
            perturbed_data = data - epsilon * sign_data_grad
            
        elif attack_type == 'pgd':
            data.requires_grad = True
            for round in range(pgd_rounds):
                print(f'Performing round {round:4,d} / {pgd_rounds:4,d} of PGD')
                data.requires_grad = True
            
                spec = torch_spectrogram(data, self.stft)
                input_sizes = torch.IntTensor([spec.size(3)]).int()
                out, output_sizes, _ = self._model(spec, input_sizes)
                out = out.transpose(0, 1)
                out = out.log_softmax(2)
                loss = criterion(out, target, output_sizes, self._target_lengths)
                
                self._model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                
                adv_sound = data - alpha * data_grad.sign() # + -> - !!!
                eta = torch.clamp(adv_sound - data_raw.data, min=-alpha, max=alpha)
                data = (data_raw + eta).detach_()
            
            perturbed_data = data            
            
        spec = torch_spectrogram(perturbed_data, self.stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes, _ = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        final_output = decoded_output[0][0]
        
        abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(data_raw.cpu().detach().numpy())**2)))
        abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(perturbed_data.cpu().detach().numpy())**2)))
        db_difference = abs_after-abs_ori
        l_distance = Levenshtein.distance(self._target_string, final_output)
        print(f"Max Decibel Difference: {db_difference:.4f}")
        print(f"Adversarial prediction: {decoded_output[0][0]}")
        print(f"Levenshtein Distance {l_distance}")
        
        torchaudio.save('audio/save.wav', src=perturbed_data.cpu(), sample_rate=16_000)
        
        return final_output