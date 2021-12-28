"""
Data preparation.

Download: https://zenodo.org/record/1117372/files/musdb18.zip
"""

import os
import csv
import logging
import sys  # noqa F401
import numpy as np
import torch
import torchaudio
from tqdm.contrib import tqdm
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)
from pathlib import Path
from shutil import copyfile

logger = logging.getLogger(__name__)
# Define where to save the csv file for generating mu vector.
# OPT_FILE = "D1_iter1_opt_xumx_output_prepare.pkl"
# XUMX_OUT_TRAIN_CSV = "D1_iter1_xumx_output_train.csv"
# XUMX_OUT_DEV_CSV = "D1_iter1_xumx_output_dev.csv"
# XUMX_OUT_TEST_CSV = "D1_iter1_xumx_output_test.csv"
SAMPLERATE = 44100

# # Restore xumx outputs
# ORIGIN_DATAPATH = Path("/mnt/md1/datasets/D1_iter1_output")
# TARGET_DATAPATH = Path("D1_iter1_output")

OPT_FILE = ""
XUMX_OUT_TRAIN_CSV = ""
XUMX_OUT_DEV_CSV = ""
XUMX_OUT_TEST_CSV = ""
ORIGIN_DATAPATH = ""
TARGET_DATAPATH = ""


def prepare_xmux_output(
    data_folder,
    save_folder,
    meta_file,
    OPT_FILE,
    XUMX_OUT_TRAIN_CSV,
    XUMX_OUT_DEV_CSV,
    XUMX_OUT_TEST_CSV,
    ORIGIN_DATAPATH,
    TARGET_DATAPATH,
    splits=["train", "valid", "test"],
    seg_dur=3.0,
    amp_th=5e-04,
    source=None,
    full_segment=False,
    split_instrument=False,
    skip_prep=False,
):
    """
    Prepares the csv files for the Voxceleb1 or Voxceleb2 datasets.
    Please follow the instructions in the README.md file for
    preparing Voxceleb2.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    splits : list
        List of splits to prepare from ['train', 'test']
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    source : str
        Path to the folder where the MUSDB dataset source is stored.
    split_instrument : bool
        Instrument-wise split
    skip_prep: Bool
        If True, skip preparation.

    Example
    -------
    >>> from recipes.mu_vector.data_prepare import prepare_xmux_outputs
    >>> data_folder = 'data'
    >>> save_folder = 'data'
    >>> splits = ['train', 'test']
    >>> prepare_xmux_outputsb(data_folder, save_folder)
    """

    # Assign global variables
    OPT_FILE = OPT_FILE
    XUMX_OUT_TRAIN_CSV = XUMX_OUT_TRAIN_CSV
    XUMX_OUT_DEV_CSV = XUMX_OUT_DEV_CSV
    XUMX_OUT_TEST_CSV = XUMX_OUT_TEST_CSV
    ORIGIN_DATAPATH = ORIGIN_DATAPATH
    TARGET_DATAPATH = TARGET_DATAPATH

    # Restore xumx outputs in terms of the datasets in speech brain
    restore_xumx_output(Path(ORIGIN_DATAPATH), Path(TARGET_DATAPATH))

    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
        "seg_dur": seg_dur,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Setting ouput files
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_csv_train = os.path.join(save_folder, XUMX_OUT_TRAIN_CSV)
    save_csv_valid = os.path.join(save_folder, XUMX_OUT_DEV_CSV)
    save_csv_test = os.path.join(save_folder, XUMX_OUT_TEST_CSV)

    msg = "\tCreating csv file for the xumx outputs Dataset.."
    logger.info(msg)

    # Split data into train and test
    wav_lst_train, wav_lst_valid, wav_lst_test = _get_sound_split_lists(
        data_folder, meta_file
    )

    # Creating csv file for train, valid, test data.
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train, save_csv_train, full_segment, amp_th
        )

    if "valid" in splits:
        prepare_csv(
            seg_dur, wav_lst_valid, save_csv_valid, full_segment, amp_th
        )

    if "test" in splits:
        prepare_csv(seg_dur, wav_lst_test, save_csv_test, full_segment, amp_th)

    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """
    Detects if the musdb data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": XUMX_OUT_TRAIN_CSV,
        "valid": XUMX_OUT_DEV_CSV,
        "test": XUMX_OUT_TEST_CSV,
    }
    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False
    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


# Used for verification split
def _get_sound_split_lists(data_folder, meta_file):
    """
    Generate xumx output lists.
    """

    train_lst = []
    valid_lst = []
    test_lst = []

    print("Getting file list...")
    # load meta data
    f = open(meta_file)
    meta_splits = list(csv.DictReader(f))

    # import random
    # meta_splits = random.sample(meta_splits,50)

    for data in meta_splits:
        if data["split"] == "train":
            train_lst.append(os.path.join(data_folder, data["ID"]))

        elif data["split"] == "valid":
            valid_lst.append(os.path.join(data_folder, data["ID"]))

        elif data["split"] == "test":
            test_lst.append(os.path.join(data_folder, data["ID"]))

    return train_lst, valid_lst, test_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

    return chunk_lst


def prepare_csv(seg_dur, wav_lst, csv_file, full_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file.

    Returns
    -------
    None
    """

    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.info(msg)

    csv_output = [["ID", "duration", "wav", "start", "stop", "inst_id"]]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        try:
            inst_id = wav_file.split("/")[-2]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue

        audio_id = my_sep.join([inst_id, wav_file.split("/")[-1]])

        # Reading the signal (to retrieve duration in seconds)
        try:
            # Stereo channels
            signal, fs = torchaudio.load(wav_file)

        except RuntimeError:
            print(wav_file)
            logger.info(f"No signal found: {wav_file}")
            continue

        if full_segment:
            audio_duration = signal.shape[-1] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[-1]

            # Composition of the csv_line
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                inst_id,
            ]
            entry.append(csv_line)

        else:
            audio_duration = signal.shape[-1] / SAMPLERATE
            # print(signal.shape,audio_duration)
            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
                if mean_sig < amp_th:
                    print(wav_file)
                    continue

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    inst_id,
                ]
                # print(csv_line)
                entry.append(csv_line)

    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = "\t%s successfully created!" % (csv_file)
    logger.info(msg)


def restore_xumx_output(origin_datapath, target_datapath):
    """
    Restore xumx output in terms of voxceleb dataset for "mu-vector" generation.
    """

    # Copy files
    print("Copy files...")
    paths = [
        p
        for p in origin_datapath.glob("*/*.wav")
        if "vocals.wav" in str(p)
        or "drums.wav" in str(p)
        or "bass.wav" in str(p)
        or "other.wav" in str(p)
    ]
    print("Restoring data in terms of voxceleb from " + str(origin_datapath))

    for src in tqdm(paths):
        instrument_id, song_id = (
            str(src).split("/")[-1].replace(".wav", ""),
            str(src).split("/")[-2],
        )

        target_dir = target_datapath / Path(instrument_id)
        target_filename = Path(song_id + ".wav")
        dst = target_dir / target_filename

        if not dst.exists():
            Path.mkdir(target_dir, parents=True, exist_ok=True)
            copyfile(src, dst)
