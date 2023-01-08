# DV2607 Project Adversarial
Attack on Speech-To-Text

# Python

This runs on `Python 3.7.12` as the code acquired from [deepspeech.pytorch](`https://github.com/SeanNaren/deepspeech.pytorch`) has some dependencies that only work before `Python 3.8`.

# Documentation

## File attack.py
Performs an adversarial attack on an audio file.

* --audio-file
> Path to the audio file to perform the attack on. Must be sampled at 16kHz. This is required.
* --target-sentence
> The target sentence for the FGSM and PGD attacks. This flag is required to pass in if you wish to use the FGSM or PGD attack methods but can be left alone if you intend to use the `untargeted` method.
* --attack-method
> Which attack method to use, possible values are { FGSM, PGD, untargeted }. Defaults to PGD
* --epsilon
> The value of epsilon to use for the attacks. Corresponds to the step size in FGSM and the maximum perturbations in PGD and untargeted attacks. Default to 0.1
* --alpha
> The value of alpha to use for the PGD and untargeted attacks. Corresponds to the step size. Defaults to 0.01
* --steps
> How many iterations to take during the PGD and untargeted attacks. Defaults to 200
* --force-download-model
> Whether to force a download of the DeepSpeech model if the program cannot find it. The relative path to the model must be `models/librispeech/librispeech_pretrained_v3.ckpt`. Defaults to False

## File stt.py

Performs basic speech-to-text using the DeepSpeech model.

* --audio-file
> Path to the audio file to perform the attack on. Must be sampled at 16kHz. This is required.
* --target-sentence
> The target sentence for the FGSM and PGD attacks. This flag is optional and if passed the program will output the Levenshtein distance between the generated text and the target-sentence.
* --original-file
> The path to the original audio file. This flag is optikonal and if passed the program will output the noise levels in decibels.
* --device
> Which device to run the program on, either `cpu` or `cuda`. Defaults to `cpu`.

## File preprocess_audio.py
Preprocesses some audio and resamples it to 16kHz so it works with DeepSpeech.

* --input-file
> Path to the audio file to perform the attack on. This is required.
* --output-file
> Path to where to save the preprocessed audio file. This is required.
## Other files

`generate_samples.py` generates a lot of audio perturbed audio files with varying parameters for the different attacks. No flags, to run with your own parameters you must change the source code.

`compare_samples.py` compares all samples generated from `generate_samples.py` and outputs all data in a fancy manner.

`graph.py` graphs original data and perturbed data. No flags, to run with your own options you must change the source code.