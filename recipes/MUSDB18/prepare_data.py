"""
The .csv preperation functions for MUSDB18.

Author
 * Y.W. Chen 2021

 """

import os
import glob
import csv
from tqdm.contrib import tqdm
from shutil import copyfile
import torchaudio
import random


def prepare_musdb18(
    datapath,
    savepath,
    origin_datapath,
    target_datapath,
    samples,
    fs=44100,
    skip_prep=False,
    data_stride=44100,
):
    """
    Prepared musdb18.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        librimix_addnoise: If True, add whamnoise to librimix datasets
    """

    if skip_prep:
        return

    else:
        assert "musdb" in datapath, "Inconsistent datapath"
        restore_musdb18(origin_datapath, target_datapath)
        create_musdb18_csv(
            datapath, savepath, samples=samples, fs=fs, data_stride=data_stride,
        )


def _get_chunks(samples, audio_id, audio_samples, data_stride, set_type):
    """
    Returns list of chunks
    """

    chunk_lst = []
    start = 0

    if set_type == "train":
        while start + samples < audio_samples:
            chunk_lst.append(
                audio_id + "_" + str(start) + "_" + str(start + samples)
            )
            start += data_stride

    else:
        while start + samples < audio_samples:
            chunk_lst.append(
                audio_id + "_" + str(start) + "_" + str(start + samples)
            )
            start += samples

    return chunk_lst


def create_musdb18_csv(
    datapath,
    savepath,
    set_types=["train", "valid", "test"],
    folder_names={
        "mixture": "mixture",
        "vocals": "vocals",
        "bass": "bass",
        "drums": "drums",
        "other": "other",
    },
    samples=44100 * 10,
    fs=44100,
    data_stride=44100,
):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in tqdm(set_types):
        mix_path = os.path.join(datapath, set_type, folder_names["mixture"])
        vocals_path = os.path.join(datapath, set_type, folder_names["vocals"])
        bass_path = os.path.join(datapath, set_type, folder_names["bass"])
        drums_path = os.path.join(datapath, set_type, folder_names["drums"])
        other_path = os.path.join(datapath, set_type, folder_names["other"])

        # import random
        # if set_type == "train":
        #     files = random.sample(os.listdir(mix_path), 1)

        # else:
        #     files = random.sample(os.listdir(mix_path), 10)

        # files = random.sample(os.listdir(mix_path), 1)
        files = os.listdir(mix_path)

        mix_fl_paths = [os.path.join(mix_path, fl) for fl in files]
        vocals_fl_paths = [os.path.join(vocals_path, fl) for fl in files]
        bass_fl_paths = [os.path.join(bass_path, fl) for fl in files]
        drums_fl_paths = [os.path.join(drums_path, fl) for fl in files]
        other_fl_paths = [os.path.join(other_path, fl) for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mean",
            "std",
            "start",
            "stop",
            "mix_wav",
            "mix_wav_format",
            "vocals_wav",
            "vocals_wav_format",
            "bass_wav",
            "bass_wav_format",
            "bass_wav_opts",
            "drums_wav",
            "drums_wav_format",
            "other_wav",
            "other_wav_format",
        ]

        if os.path.exists(savepath + "/musdb18_" + set_type + ".csv"):
            print("/musdb18_" + set_type + ".csv exists!")

        else:
            with open(
                savepath + "/musdb18_" + set_type + ".csv", "w"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                dataset = []
                for (
                    i,
                    (mix_path, vocals_path, bass_path, drums_path, other_path),
                ) in enumerate(
                    zip(
                        mix_fl_paths,
                        vocals_fl_paths,
                        bass_fl_paths,
                        drums_fl_paths,
                        other_fl_paths,
                    )
                ):

                    audio_id = mix_path.split("/")[-1]
                    signal, fs = torchaudio.load(mix_path)
                    wav = signal.mean(0)
                    mean = wav.mean().item()
                    std = wav.std().item()

                    audio_samples = signal.shape[-1]
                    audio_duration = audio_samples / fs

                    # Chunk by samples
                    uniq_chunks_list = _get_chunks(
                        samples, audio_id, audio_samples, data_stride, set_type
                    )

                    uniq_chunks_list.append(
                        audio_id
                        + "_"
                        + str(audio_samples - samples)
                        + "_"
                        + str(audio_samples)
                    )

                    for j in range(len(uniq_chunks_list)):
                        s, e = uniq_chunks_list[j].split("_")[-2:]
                        start_sample = int(s)
                        end_sample = int(e)

                        # Append the last segment < seg_dur
                        if j == len(uniq_chunks_list) - 1:
                            end_sample = signal.shape[-1]
                            start_sample = (
                                end_sample - samples
                            )  # For the same segment duration

                        row = {
                            "ID": str(i) + "_" + str(j),
                            "duration": str(audio_duration),
                            "mean": mean,
                            "std": std,
                            "start": start_sample,
                            "stop": end_sample,
                            "mix_wav": mix_path,
                            "mix_wav_format": "wav",
                            "vocals_wav": vocals_path,
                            "vocals_wav_format": "wav",
                            "bass_wav": bass_path,
                            "bass_wav_format": "wav",
                            "drums_wav": drums_path,
                            "drums_wav_format": "wav",
                            "other_wav": other_path,
                            "other_wav_format": "wav",
                        }

                        dataset.append(row)

                SEED = 44
                random.seed(SEED)
                random.shuffle(dataset)

                # Shuffle
                for data in dataset:
                    writer.writerow(data)


def restore_musdb18(origin_datapath, target_datapath):
    """
    Restore musdb18 for musdb18 separation.
    """
    validation_tracks = [
        "Actions - One Minute Smile",
        "Clara Berry And Wooldog - Waltz For My Victims",
        "Johnny Lokke - Promises & Lies",
        "Patrick Talbot - A Reason To Leave",
        "Triviul - Angelsaint",
        "Alexander Ross - Goodbye Bolero",
        "Fergessen - Nos Palpitants",
        "Leaf - Summerghost",
        "Skelpolu - Human Mistakes",
        "Young Griffo - Pennies",
        "ANiMAL - Rockshow",
        "James May - On The Line",
        "Meaxic - Take A Step",
        "Traffic Experiment - Sirens",
    ]

    # Copy files
    print("Copy files...")
    paths = [
        p
        for p in glob.glob(
            os.path.join(origin_datapath, "**/*.wav"), recursive=True
        )
        if not p.endswith("accompaniment.wav")
        and not p.endswith("linear_mixture.wav")
    ]

    print(
        "Restoring data in terms of speech brain from " + str(origin_datapath)
    )

    for src in tqdm(paths):
        instrument_id, song_id = (
            str(src).split("/")[-1].replace(".wav", ""),
            str(src).split("/")[-2],
        )

        split = "train" if "train" in str(src) else "test"
        if song_id in validation_tracks:
            split = "valid"

        target_dir = os.path.join(target_datapath, split, instrument_id)
        target_filename = song_id + ".wav"
        dst = os.path.join(target_dir, target_filename)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if not os.path.exists(dst):
            copyfile(src, dst)
