"""
Data preparation.

Download: https://magenta.tensorflow.org/datasets/nsynth
"""

import os
import csv
import json
import logging
import glob
import sys  # noqa F401
import numpy as np
import torch
import torchaudio
from tqdm.contrib import tqdm
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_nsynth_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000
META = "meta"


def prepare_nsynth(
    data_folder,
    save_folder,
    verification_pairs_file,
    meta_train,
    meta_valid,
    meta_test,
    splits=["train", "dev", "test"],
    seg_dur=3.0,
    amp_th=5e-04,
    source=None,
    split_instrument=False,
    random_segment=False,
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
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    source : str
        Path to the folder where the Nsynth dataset source is stored.
    split_instrument : bool
        Instrument-wise split
    random_segment : bool
        Train random segments
    skip_prep: Bool
        If True, skip preparation.

    Example
    -------
    >>> from recipes.mu_vector.data_prepare import prepare_nsynth
    >>> data_folder = 'data'
    >>> save_folder = 'data'
    >>> splits = ['train', 'dev', 'test']
    >>> prepare_nsynth(data_folder, save_folder, splits)
    """

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "save_folder": save_folder,
        "seg_dur": seg_dur,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_dev = os.path.join(save_folder, DEV_CSV)

    # Create the data folder contains test data from the source
    if source is not None:
        logger.info("Please Extract Nsynth from original datadir first.")
        return

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains VoxCeleb data
    if "," in data_folder:
        data_folder = data_folder.replace(" ", "").split(",")
    else:
        data_folder = [data_folder]

    # _check_nsynth_folders(data_folder, splits)

    msg = "\tCreating csv file for the Nsynth Dataset.."
    logger.info(msg)

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev = _get_sound_split_lists(
        data_folder, meta_train, meta_valid
    )

    # Creating csv file for training data
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train, save_csv_train, random_segment, amp_th
        )

    if "dev" in splits:
        prepare_csv(seg_dur, wav_lst_dev, save_csv_dev, random_segment, amp_th)

    # For PLDA verification or test on instrument family classification
    if "test" in splits:
        prepare_csv_enrol_test(
            data_folder, save_folder, verification_pairs_file
        )

    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """
    Detects if the voxceleb data_preparation has been already done.
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
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
        "test": TEST_CSV,
        "enrol": ENROL_CSV,
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


def _check_nsynth_folders(data_folders, splits):
    """
    Check if the data folder actually contains the Nsynth dataset.

    If it does not, raise an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
    """
    for data_folder in data_folders:

        if "train" in splits:
            folder_nsynth = os.path.join(
                data_folder, "wav", "keyboard_electronic_070"
            )
            if not os.path.exists(folder_nsynth):
                err_msg = "the specified folder does not contain Nsynth"
                raise FileNotFoundError(err_msg)

        if "test" in splits:
            folder = os.path.join(data_folder, "wav", "vocal_synthetic_000")
            if not os.path.exists(folder):
                err_msg = (
                    "the folder %s does not exist (as it is expected in "
                    "the nsynth dataset)" % folder
                )
                raise FileNotFoundError(err_msg)

        folder = os.path.join(data_folder, "meta")
        if not os.path.exists(folder):
            err_msg = (
                "the folder %s does not exist (as it is expected in "
                "the nsynth dataset)" % folder
            )
            raise FileNotFoundError(err_msg)


# Used for verification split
def _get_sound_split_lists(data_folders, meta_train, meta_valid):
    """
    Tot. number of nsynths = 1006.
    Splits the audio file list into train and dev.
    This function automatically removes verification test files from the training and dev set (if any).
    """
    train_lst = []
    dev_lst = []

    print("Getting file list...")
    for data_folder in data_folders:

        train_snts = []
        dev_snts = []

        # load meta data
        f = open(meta_train)
        train_insts = json.load(f)
        train_insts = list(train_insts.keys())

        f = open(meta_valid)
        valid_insts = json.load(f)
        valid_insts = list(valid_insts.keys())

        path = os.path.join(data_folder, "wav", "**", "*.wav")
        files = [file for file in glob.glob(path, recursive=True)]
        # import random
        # files = random.sample(files,500)

        # avoid test speakers for train and dev splits
        for f in tqdm(files):
            try:
                sound_id = (
                    f.split("/wav/")[1].split("/")[0]
                    + "-"
                    + f.split("/wav/")[1].split("/")[1]
                )
                sound_id = sound_id.replace(".wav", "")
            except ValueError:
                logger.info(f"Malformed path: {f}")
                continue
            if sound_id in train_insts:
                train_snts.append(f)
            elif sound_id in valid_insts:
                dev_snts.append(f)

        train_lst.extend(train_snts)
        dev_lst.extend(dev_snts)

    return train_lst, dev_lst


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


def prepare_csv(seg_dur, wav_lst, csv_file, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    random_segment: bool
        Read random segments
    amp_th: float
        Threshold on the average amplitude on the chunk.
        If under this threshold, the chunk is discarded.

    Returns
    -------
    None
    """

    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.info(msg)

    csv_output = [
        ["ID", "duration", "wav", "start", "stop", "inst_id", "inst_family"]
    ]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        try:
            inst_id = wav_file.split("/")[-2]
            pitch, velocity = wav_file.split("/")[-1].split("-")
            inst_family = inst_id.split("_")[0]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue
        audio_id = my_sep.join([inst_id, pitch, velocity.split(".")[0]])

        # Reading the signal (to retrieve duration in seconds)
        signal, fs = torchaudio.load(wav_file)
        signal = signal.squeeze(0)

        if random_segment:
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]

            # Composition of the csv_line
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                inst_id,
                inst_family,
            ]
            entry.append(csv_line)
        else:
            audio_duration = signal.shape[0] / SAMPLERATE

            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
                if mean_sig < amp_th:
                    continue

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    inst_id,
                    inst_family,
                ]
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


def prepare_csv_enrol_test(data_folders, save_folder, verification_pairs_file):
    """
    Creates the csv file for test data (useful for verification)

    Arguments
    ---------
    data_folder : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.

    Returns
    -------
    None
    """

    # msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    # logger.debug(msg)

    csv_output_head = [
        ["ID", "duration", "wav", "start", "stop", "inst_id", "inst_family"]
    ]  # noqa E231

    for data_folder in data_folders:

        test_lst_file = verification_pairs_file

        enrol_ids, test_ids = [], []

        # Get unique ids (enrol and test sound)
        for line in open(test_lst_file):
            e_id = line.split(" ")[1].rstrip().split(".")[0].strip()
            t_id = line.split(" ")[2].rstrip().split(".")[0].strip()
            enrol_ids.append(e_id)
            test_ids.append(t_id)

        enrol_ids = list(np.unique(np.array(enrol_ids)))
        test_ids = list(np.unique(np.array(test_ids)))

        # Prepare enrol csv
        logger.info("preparing enrol csv")
        enrol_csv = []
        for id in enrol_ids:
            wav = data_folder + "/wav/" + id + ".wav"

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            inst_id = wav.split("/")[-2]
            pitch, velocity = wav.split("/")[-1].split("-")
            inst_family = inst_id.split("_")[0]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                inst_id,
                inst_family,
            ]

            enrol_csv.append(csv_line)

        csv_output = csv_output_head + enrol_csv
        csv_file = os.path.join(save_folder, ENROL_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)

        # Prepare test csv
        logger.info("preparing test csv")
        test_csv = []
        for id in test_ids:
            wav = data_folder + "/wav/" + id + ".wav"

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            inst_id = wav.split("/")[-2]
            pitch, velocity = wav.split("/")[-1].split("-")
            inst_family = inst_id.split("_")[0]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                inst_id,
                inst_family,
            ]

            test_csv.append(csv_line)

        csv_output = csv_output_head + test_csv
        csv_file = os.path.join(save_folder, TEST_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)
