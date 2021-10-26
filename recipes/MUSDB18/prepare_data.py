"""
The .csv preperation functions for MUSDB18.

Author
 * Cem Subakan 2020

 """

import os
import glob
import csv
from tqdm.contrib import tqdm
from shutil import copyfile
import torchaudio


SAMPLERATE = 44100


def prepare_musdb18(
    datapath,
    savepath,
    origin_datapath,
    target_datapath,
    train_seg_dur=3.0,
    valid_seg_dur=40.0,
    fs=44100,
    skip_prep=False,
    librimix_addnoise=False,
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
            datapath,
            savepath,
            train_seg_dur=train_seg_dur,
            valid_seg_dur=valid_seg_dur,
            fs=fs,
        )


def _get_chunks(seg_dur, audio_id, audio_duration, num_chunks):
    """
    Returns list of chunks
    """
    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

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
    train_seg_dur=3.0,
    valid_seg_dur=40.0,
    fs=44100,
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

        seg_dur = train_seg_dur if set_type == "train" else valid_seg_dur

        # import random
        # if set_type == "train":
        # files = random.sample(os.listdir(mix_path),1)

        # else:
        files = os.listdir(mix_path)

        mix_fl_paths = [os.path.join(mix_path, fl) for fl in files]
        vocals_fl_paths = [os.path.join(vocals_path, fl) for fl in files]
        bass_fl_paths = [os.path.join(bass_path, fl) for fl in files]
        drums_fl_paths = [os.path.join(drums_path, fl) for fl in files]
        other_fl_paths = [os.path.join(other_path, fl) for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "start",
            "stop",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "vocals_wav",
            "vocals_wav_format",
            "vocals_wav_opts",
            "bass_wav",
            "bass_wav_format",
            "bass_wav_opts",
            "drums_wav",
            "drums_wav_format",
            "drums_wav_opts",
            "other_wav",
            "other_wav_format",
            "other_wav_opts",
        ]

        if os.path.exists(savepath + "/musdb18_" + set_type + ".csv"):
            print("/musdb18_" + set_type + ".csv exists!")

        else:
            with open(
                savepath + "/musdb18_" + set_type + ".csv", "w"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
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
                    audio_duration = signal.shape[-1] / fs
                    num_chunks = int(
                        audio_duration / seg_dur
                    )  # all in milliseconds

                    uniq_chunks_list = _get_chunks(
                        seg_dur, audio_id, audio_duration, num_chunks
                    )
                    uniq_chunks_list.append(
                        audio_id
                        + "_"
                        + str(num_chunks * seg_dur)
                        + "_"
                        + str(audio_duration)
                    )

                    for j in range(len(uniq_chunks_list)):
                        s, e = uniq_chunks_list[j].split("_")[-2:]
                        start_sample = int(float(s) * fs)
                        end_sample = int(float(e) * fs)

                        # Append the last segment < seg_dur
                        if j == len(uniq_chunks_list):
                            end_sample = signal.shape[-1][0]

                        row = {
                            "ID": str(i) + "_" + str(j),
                            "duration": str(audio_duration),
                            "start": start_sample,
                            "stop": end_sample,
                            "mix_wav": mix_path,
                            "mix_wav_format": "wav",
                            "mix_wav_opts": None,
                            "vocals_wav": vocals_path,
                            "vocals_wav_format": "wav",
                            "vocals_wav_opts": None,
                            "bass_wav": bass_path,
                            "bass_wav_format": "wav",
                            "bass_wav_opts": None,
                            "drums_wav": drums_path,
                            "drums_wav_format": "wav",
                            "drums_wav_opts": None,
                            "other_wav": other_path,
                            "other_wav_format": "wav",
                            "other_wav_opts": None,
                        }

                        writer.writerow(row)


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
