"""
Data preparation.
Download: http://www.openslr.org/12
Author
------
Victor Chen, 2022
"""

from signal import signal
import DALI as dali_code
import os
import csv
import logging
import torchaudio
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)
import re

logger = logging.getLogger(__name__)
OPT_FILE = "opt_data_prepare.pkl"
SAMPLERATE = 16000


def prepare_data(
    text_data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
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
    splits = [tr_splits] + [dev_splits] + [te_splits]
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

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

    # Create csv for train, valid, test, splits
    for split in splits:
        create_csv(
            text_data_folder,
            save_folder,
            split,
            select_n_sentences,
        )

    # saving options
    save_pkl(conf, save_opt)


def text_normalization(wrds):
    from sacremoses import MosesPunctNormalizer

    mpn = MosesPunctNormalizer()
    exception = "'"
    
    wrds_splits = wrds.split(" ")

    # Upper only for content text
    for i in range(len(wrds_splits)):
        if "<UNK>" in wrds_splits[i]:
            wrds_splits[i] = wrds_splits[i].lower()

        elif "<silence>" not in wrds_splits[i] and "<music>" not in wrds_splits[i]:
            wrds_splits[i] = mpn.normalize(wrds_splits[i]).upper()
            wrds_splits[i] = re.sub(r'[^\w'+exception+']', ' ', wrds_splits[i])

    wrds = " ".join(
            [str(elem) for elem in wrds_splits]
        )

    wrds = " ".join(wrds.split())
    wrds = wrds.strip()

    return wrds


def create_csv(
    text_data_folder, save_folder, splits, select_n_sentences
):

    csv_name = splits[1].split('_')[0]
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, csv_name + ".csv")

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "wrd"]]
    snt_cnt = 0

    for split in splits:
        text_path = os.path.join(text_data_folder, split, "text")
        text_lst = open(text_path, "r").read().split("\n")

        # Processing all the wav files in wav_lst
        for text_file in text_lst:

            if text_file:

                snt_id = text_file.split(" ")[0]
                wrds = " ".join(
                    [str(elem) for elem in text_file.split(" ")[1:]]
                )

                wrds = text_normalization(wrds)

                csv_line = [
                    snt_id,
                    str(" ".join(wrds.split("_"))),
                ]

                #  Appending current file to the csv_lines list
                csv_lines.append(csv_line)
                snt_cnt = snt_cnt + 1

                if snt_cnt == select_n_sentences:
                    break

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
        split = split[1].split('_')[0]
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