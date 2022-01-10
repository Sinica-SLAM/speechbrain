"""
Data preparation.

Download: http://www.openslr.org/12

Author
------
Mirco Ravanelli, Ju-Chieh Chou, Loren Lugosch 2020
"""

import os
import csv
import logging
import torchaudio
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_data_prepare.pkl"
SAMPLERATE = 16000


def prepare_data(
    text_data_folder,
    audio_data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    create_lexicon=False,
    skip_prep=False,
):
    """
    This class prepares the csv files for the LibriSpeech dataset.
    Download link: http://www.openslr.org/12

    Arguments
    ---------
    audio_data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    tr_splits : list
        List of train splits to prepare from ['test-others','train-clean-100',
        'train-clean-360','train-other-500'].
    dev_splits : list
        List of dev splits to prepare from ['dev-clean','dev-others'].
    te_splits : list
        List of test splits to prepare from ['test-clean','test-others'].
    save_folder : str
        The directory where to store the csv files.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of librispeech splits (e.g, train-clean, train-clean-360,..) to
        merge in a singe csv file.
    merge_name: str
        Name of the merged csv file.
    create_lexicon: bool
        If True, it outputs csv files containing mapping between grapheme
        to phonemes. Use it for training a G2P system.
    skip_prep: bool
        If True, data preparation is skipped.


    Example
    -------
    >>> audio_data_folder = 'datasets/LibriSpeech'
    >>> tr_splits = ['train-clean-100']
    >>> dev_splits = ['dev-clean']
    >>> te_splits = ['test-clean']
    >>> save_folder = 'librispeech_prepared'
    >>> prepare_librispeech(audio_data_folder, save_folder, tr_splits, dev_splits, te_splits)
    """

    if skip_prep:
        return
    audio_data_folder = audio_data_folder
    splits = tr_splits + dev_splits + te_splits
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Restore and convert data from mp3 to wav
    logger.info("Data restoring...")
    restore_data(audio_data_folder)

    # create csv files for each split
    for split_index in range(len(splits)):
        split = splits[split_index]

        create_csv(
            audio_data_folder,
            text_data_folder,
            save_folder,
            split,
            select_n_sentences,
        )

    # saving options
    save_pkl(conf, save_opt)


def restore_data(audio_data_folder):

    for filename in os.listdir(audio_data_folder):
        if filename.endswith(".mp3"):  # or .avi, .mpeg, whatever.
            save_path = os.path.join(audio_data_folder, filename)
            os.system(
                "ffmpeg -hide_banner -y -i "
                + save_path
                + " -ar 16000 -ac 1 -f wav "
                + save_path.replace("mp3", "wav")
            )
        else:
            continue


def create_csv(
    audio_data_folder, text_data_folder, save_folder, split, select_n_sentences
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "start", "stop", "wav", "spk_id", "wrd"]]

    snt_cnt = 0

    if split != "dali_talt":
        seg_path = os.path.join(text_data_folder, split, "segments")
        text_path = os.path.join(text_data_folder, split, "text")

        seg_lst = open(seg_path, "r").read().split("\n")
        text_lst = open(text_path, "r").read().split("\n")

        # Processing all the wav files in wav_lst
        for seg_file, text_file in zip(seg_lst, text_lst):

            if seg_file and text_file:

                snt_id = seg_file.split(" ")[0]
                spk_id = seg_file.split(" ")[1]
                wrds = " ".join(
                    [str(elem) for elem in text_file.split(" ")[1:]]
                )

                start = seg_file.split(" ")[-2]
                stop = seg_file.split(" ")[-1]

                duration = float(stop) - float(start)
                wav_file = os.path.join(
                    audio_data_folder, spk_id.split("-")[-1] + ".wav"
                )

                csv_line = [
                    snt_id,
                    str(duration),
                    start,
                    stop,
                    wav_file,
                    spk_id,
                    str(" ".join(wrds.split("_"))),
                ]

                #  Appending current file to the csv_lines list
                csv_lines.append(csv_line)
                snt_cnt = snt_cnt + 1

                if snt_cnt == select_n_sentences:
                    break

    else:
        csv_lines = [["spk_id", "duration", "wav", "wrd"]]
        text_path = os.path.join(text_data_folder, split, "text")
        text_lst = open(text_path, "r").read().split("\n")

        for text_file in text_lst:
            if text_file:
                spk_id = text_file.split(" ")[0]
                wrds = " ".join(
                    [str(elem) for elem in text_file.split(" ")[1:]]
                )

                wav_file = os.path.join(
                    audio_data_folder, spk_id.split("-")[-1] + ".wav"
                )

                signal, fs = torchaudio.load(wav_file)
                print(wav_file)
                signal = signal.squeeze(0)
                duration = signal.shape[0] / SAMPLERATE

                csv_line = [
                    spk_id,
                    duration,
                    wav_file,
                    str(" ".join(wrds.split("_"))),
                ]

                #  Appending current file to the csv_lines list
                csv_lines.append(csv_line)

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def skip(splits, save_folder, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
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
