import os 
import sys
import csv
import random
import shutil
from tqdm import tqdm

dataset_dir = '/mnt/md0/user_fal1210/Corpora_30/wsj0-mix/2speakers'
small_dataset_dir = '/mnt/md0/user_fal1210/corpora/wsj0-mix_small/2speakers'
split = 1/10

os.makedirs(small_dataset_dir, exist_ok = True)


for mode in ['tr', 'cv', 'tt']:
    mode_dir = os.path.join(dataset_dir, 'wav8k', 'min', mode)
    small_mode_dir = os.path.join(small_dataset_dir, 'wav8k', 'min', mode)
    os.makedirs(small_mode_dir, exist_ok = True)
    wavs = os.listdir(os.path.join(mode_dir, 'mix'))
    wav_num = len(wavs)
    random_wavs = random.sample(wavs, wav_num // 10)
    for wav in tqdm(random_wavs):
        for wav_type in ['mix', 'mix_both', 'mix_single', 's1', 's2', 'noise']:
            source = os.path.join(mode_dir, wav_type, wav)
            small_wav_dir = os.path.join(small_mode_dir, wav_type)
            os.makedirs(small_wav_dir, exist_ok=True)
            destination = os.path.join(small_wav_dir, wav)
            shutil.copyfile(source, destination)
            