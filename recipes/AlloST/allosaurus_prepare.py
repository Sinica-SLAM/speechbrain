"""
Data preparation for phone sequence

Author
-----
YAO-FEI, CHENG 2021
"""

import logging
import os
import json
import subprocess

from typing import Any, Dict
from multiprocessing import Pool

from tqdm import tqdm
from allosaurus.app import read_recognizer

ALLOSAURUS = read_recognizer()
logger = logging.getLogger(__name__)


def read_json(json_path: str) -> Dict[str, Any]:
    """Read the given json file, and return a dictionary"""
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data


def save_json(json_path: str, data: Dict[str, Any]):
    """Save the given data as json"""
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)


def change_file_encoding(wav_path: str, save_folder: str):
    """Change file encoding from 32 bit floating to 16 bit integer"""
    filename = wav_path.split("/")[-1]
    command = f"sox -v 0.99 {wav_path} -b 16 {save_folder}/{filename}".split(
        " "
    )
    subprocess.run(command)


def recognize_all_phone(
    wav_folder: str,
    phone_folder: str,
    data_json: Dict[str, any],
    number_of_workers: int = 64,
) -> Dict[str, any]:
    """Recognize phone posterior from all of systems"""
    wav_ids = list(data_json.keys())
    phone_wav_folder = f"{phone_folder}/wav"

    if not os.path.exists(phone_wav_folder):
        os.mkdir(phone_wav_folder)

    # TODO (jamfly): turn audio file from 32 bit floating encoding to 16 bit integer
    # because allosaurus do not support 32 bit floating, fix it when they update the model
    # ref: https://github.com/xinjli/allosaurus/issues/33
    data_to_be_change = list(
        map(lambda wav_id: f"{wav_folder}/{wav_id}.wav", wav_ids)
    )
    data_to_be_change = list(
        map(lambda data: (data, phone_wav_folder), data_to_be_change)
    )
    with Pool(processes=number_of_workers) as pool:
        pool.starmap(change_file_encoding, data_to_be_change)

    phone_lexicon = set()

    # recognize phone labels
    for wav_id in tqdm(wav_ids):
        phone = ALLOSAURUS.recognize(f"{phone_folder}/wav/{wav_id}.wav")

        phone_sequence = phone.split(" ")
        phone_lexicon.update(phone_sequence)

        data_json[wav_id]["allosaurus"] = phone

    # since some of phone sequence are composed in two labels
    base = ord("\U00013000")
    phone_lexicon = sorted(phone_lexicon)
    phone_lexicon = list(phone_lexicon)

    for wav_id in tqdm(wav_ids):
        original_phone = data_json[wav_id]["allosaurus"]
        sequenced_phone = []

        for phone in original_phone.split(" "):
            order_in_lexicon = phone_lexicon.index(phone)
            label = chr(base + order_in_lexicon)
            sequenced_phone.append(label)

        sequenced_phone = "".join(sequenced_phone)
        data_json[wav_id]["allosaurus_for_bpe"] = sequenced_phone

    return data_json


def prepare_allosaurus(save_folder: str, number_of_workers: int = 64):
    """
    Prepare the json files for the phone for Fisher Spanish English Corpus.
    Arguments
    ---------
    save_folder: str:
        Path of train/valid/test specification file will be saved.
    """
    datasets = ["dev", "dev2", "test", "train"]

    progress_bar = tqdm(datasets)
    for dataset in progress_bar:
        phone_folder = f"{save_folder}/{dataset}/phone"
        if not os.path.exists(phone_folder):
            os.mkdir(phone_folder)

        progress_bar.set_description(dataset)

        json_path = f"{save_folder}/{dataset}/data.json"
        data_json = read_json(json_path=json_path)

        wav_folder = f"{save_folder}/{dataset}/wav"

        data_json = recognize_all_phone(
            wav_folder=wav_folder,
            phone_folder=phone_folder,
            data_json=data_json,
            number_of_workers=number_of_workers,
        )

        save_json(json_path=json_path, data=data_json)
