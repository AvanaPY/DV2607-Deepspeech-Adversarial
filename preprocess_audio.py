import librosa    
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='Original wav file', dest='input_file')
    parser.add_argument('--output-file', type=str, help='Output wav file', dest='output_file')
    args = parser.parse_args()

    wav_file = args.input_file
    if wav_file[-4:] == ".wav":
        print(f"Convert {wav_file}")
        y, s = librosa.load(wav_file, sr=16000)
        librosa.output.write_wav(args.output_file, y, s)