import os 
import sys
import csv
import random
import shutil
from tqdm import tqdm

dataset_dir = '/mnt/md0/user_fal1210/corpora_135/TAT-mix'
small_dataset_dir = '/mnt/md0/user_fal1210/corpora_135/TAT-mix_small'
split = 1/100

os.makedirs(small_dataset_dir, exist_ok = True)

random_wavs = {'tr':[], 'cv':[], 'tt':[]}
for channel in ['android', 'condenser', 'ios', 'lavalier', 'XYH-6-X', 'XYH-6-Y']:
    for mode in ['tr', 'cv', 'tt']:
        mode_dir = os.path.join(dataset_dir, f'wav8k_{channel}', 'min', mode)
        small_mode_dir = os.path.join(small_dataset_dir, f'wav8k_{channel}', 'min', mode)
        os.makedirs(small_mode_dir, exist_ok = True)
        if len(random_wavs[mode]) == 0:
            wavs = os.listdir(os.path.join(mode_dir, 's1'))
            wav_num = len(wavs)
            random_wavs[mode] = random.sample(wavs, int(wav_num * split))
        for wav in tqdm(random_wavs[mode]):
            for wav_type in ['mix_100', 's1', 's2']:
                source = os.path.join(mode_dir, wav_type, wav)
                small_wav_dir = os.path.join(small_mode_dir, wav_type)
                os.makedirs(small_wav_dir, exist_ok=True)
                destination = os.path.join(small_wav_dir, wav)
                shutil.copyfile(source, destination)