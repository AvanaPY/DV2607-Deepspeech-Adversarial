import torch
import torch.nn as nn
import numpy as np
import Levenshtein
import torchaudio

from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.configs.inference_config import TranscribeConfig

from stft import STFT, magphase
from model_downloader import verify_model_exist

SAMPLE_RATE = 16_000

def target_str_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for word in sentence:
        out.append(labels.index(word))
    return torch.IntTensor(out)

def torch_spectrogram(sound, stft):
    real, imag = stft(sound)
    mag, cos, sin = magphase(real, imag)
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
                 save_path : str = 'save.wav'):
        self._device = 'cuda'
        self._attack_method = attack_method
        self._model = model
        self._model.to(self._device)
        self._model.train()              # Set the model into train mode so we can use backward() on the RNN
        self._decoder = decoder
        self._sound = sound
        self._target_string = target_str
        self._target = target_str_to_label(target_str)
        self._target = self._target.view(1,-1)
        self._target_lengths = torch.IntTensor([self._target.shape[1]]).view(1,-1)
        self._save_path = save_path
        
        n_fft       = int(SAMPLE_RATE * 0.02)
        hop_length  = int(SAMPLE_RATE * 0.01)
        win_length  = int(SAMPLE_RATE * 0.01)
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
    
    def attack(self, epsilon : float = 0.1, alpha : float = 1e-2, pgd_rounds : int = 50):
        
        data, target = self._sound.to(self.device), self._target.to(self.device)
        data_raw = data.clone().detach()
        
        criterion = nn.CTCLoss()
        
        spec = torch_spectrogram(data, self.stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes, _ = self._model(spec, input_sizes)
        decoded_output, decoded_offsets = self._decoder.decode(out, output_sizes)
        original_output = decoded_output[0][0]
        
        # Let us perform fgsm fuck you
        attack_type = self._attack_method
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
                done = int(50 * round / pgd_rounds)
                print(f'\rPGD [{"="*done}{" "*(50-done)}] {round:5,d} /{pgd_rounds:5,d}', end='')
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
            print()       
            
        spec = torch_spectrogram(perturbed_data, self.stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes, _ = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        final_output = decoded_output[0][0]
        
        abs_ori = np.log10(np.max(np.absolute(data_raw.cpu().detach().numpy())))
        abs_after = np.log10(np.max(np.absolute(perturbed_data.cpu().detach().numpy())))
        db_difference = 20 * (abs_after-abs_ori)
        l_distance = Levenshtein.distance(self._target_string, final_output)
        
        if self._save_path:
            torchaudio.save(self._save_path, src=perturbed_data.cpu(), sample_rate=SAMPLE_RATE)
        
        return original_output, final_output, db_difference, l_distance