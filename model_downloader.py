from typing import *
import os 
import urllib.request
import requests
import time

def verify_model_exist(model_path : str, model_url : Optional[str] = None, force_download : bool = False):
    if force_download or not os.path.exists(model_path):
        download_model(model_path, model_url)
    else:
        print(f'Found model at {model_path}')  
    
def download_model(model_path : str, model_url : str):
    if model_url is None:
        raise ValueError('`model_url` cannot be None')
    
    dirname, file_name = os.path.split(model_path)
    
    print(f'Downloading model to {dirname}')
    start_time = time.time()
    with open(model_path, 'wb') as f:
        response = requests.get(model_url, stream=True)
        total_length = response.headers.get('content-length')
        
        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                
                c_time = time.time()
                d_time = c_time - start_time
                progress_per_sec = (dl / d_time) * 1e-6 * 8    # Mb per seconds
                print(f'\r> {dl * 1e-6 * 8:6.1f} / {total_length * 1e-6 * 8:.1f} [{"="*done}{" "*(50 - done)}] ({progress_per_sec:.1f}Mb/s) {file_name}', end='')
            print()