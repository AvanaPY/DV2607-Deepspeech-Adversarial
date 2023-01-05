import os
import sys
sys.path.append("deepspeech.pytorch/")
from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.configs.inference_config import TranscribeConfig
import torchaudio
from attacker import Attacker
model_path = '/home/emkl/Documents/School/DV2607/ds2aa/models/librispeech/librispeech_pretrained_v3.ckpt'
audio_path = '/home/emkl/Documents/School/DV2607/ds2aa/audio16k/harvard.wav'

target_sentence_str = 'I AM A BIG FAN OF COOKIES AND I WOULD LIKE IF YOU GAVE ME SOME ORANGES'

cfg = TranscribeConfig
model = load_model(device="cpu", model_path=model_path)
decoder = load_decoder(labels=model.labels, cfg=cfg.lm)

sound, sample_rate = torchaudio.load(audio_path)
target_sentence = target_sentence_str.upper()

attacker = Attacker(model, decoder, sound, target_sentence)
output = attacker.attack()