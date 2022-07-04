import os 
import sys
import csv
import random
import shutil
from tqdm import tqdm

dataset_dir = '/mnt/md0/user_fal1210/corpora/DPFN_wham_sb'
csv_path = '/mnt/md0/user_fal1210/SLAM/speechbrain/recipes/WHAMandWHAMR/separation/results/dprnn-wham/1234/save'
small_dataset_dir = dataset_dir + '_small'
split = 1/10

os.makedirs(small_dataset_dir, exist_ok = True)


for mode in ['tr', 'cv', 'tt']:
    csv_file = os.path.join(csv_path, f'whamorg_{mode}.csv')
    csv_rows = []
    with open(csv_file, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            csv_rows.append(row)
            
    mode_dir = os.path.join(dataset_dir, f"examples_{mode}")
    small_mode_dir = os.path.join(small_dataset_dir, f"examples_{mode}")
    os.makedirs(small_mode_dir, exist_ok = True)
    wavs = os.listdir(mode_dir)
    wav_num = len(wavs) // 5
    random_ids = random.sample(range(0, wav_num), wav_num // 10)
    for i in tqdm(range(len(random_ids))):
        for wav_type in ['mix', 'source1', 'source1hat', 'source2', 'source2hat']:
            source = os.path.join(mode_dir, f'item{random_ids[i]}_{wav_type}.wav')
            destination = os.path.join(small_mode_dir, f'item{i}_{wav_type}.wav')
            shutil.copyfile(source, destination)
            
    csv_columns = [
        "ID",
        "duration",
        "mix_wav",
        "mix_wav_format",
        "mix_wav_opts",
        "s1_wav",
        "s1_wav_format",
        "s1_wav_opts",
        "s2_wav",
        "s2_wav_format",
        "s2_wav_opts",
        "noise_wav",
        "noise_wav_format",
        "noise_wav_opts",
    ]

    with open(small_dataset_dir + "/whamorg_" + mode + ".csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i in tqdm(range(len(random_ids))):
            csv_row = csv_rows[random_ids[i]]
            row = {
                "ID": i,
                "duration": csv_row[1],
                "mix_wav": csv_row[2],
                "mix_wav_format": "wav",
                "mix_wav_opts": None,
                "s1_wav": csv_row[5],
                "s1_wav_format": "wav",
                "s1_wav_opts": None,
                "s2_wav": csv_row[10],
                "s2_wav_format": "wav",
                "s2_wav_opts": None,
                "noise_wav": csv_row[11],
                "noise_wav_format": "wav",
                "noise_wav_opts": None,
            }
            writer.writerow(row)