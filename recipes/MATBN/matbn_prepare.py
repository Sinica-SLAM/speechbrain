import logging
import os
from dataclasses import dataclass, is_dataclass, asdict
from typing import Dict

import re
import json

logger = logging.getLogger(__name__)


@dataclass
class Data:
    wav: str
    transcription: str


class DataClassJSONEncoder(json.JSONEncoder):
    def default(self, object):
        if is_dataclass(object):
            return asdict(object)
        return super().default(object)


def prepare_matbn(
    dataset_folder: str,
    save_folder: str,
    keep_unk: bool = False,
    skip_prep: bool = False,
):
    if skip_prep:
        return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    splits = ["eval", "train"]  # dev, test

    for split in splits:
        split_folder = os.path.join(dataset_folder, split)
        wav_folder = os.path.join(split_folder, "wav")
        data_folder = os.path.join(split_folder, "data")
        if check_folders_exist(wav_folder, data_folder) is not True:
            logger.error(
                "the folder wav or data does not exist (it is expected in the "
                "MATBN dataset)"
            )

        text_path = os.path.join(data_folder, "text")
        data = extract_data(text_path, wav_folder)

        useful_data = remove_useless_data(data, keep_unk)

        save_path = os.path.join(save_folder, f"{split}.json")

        with open(save_path, "w", encoding="utf-8") as save_file:
            json.dump(
                useful_data,
                save_file,
                indent=2,
                ensure_ascii=False,
                cls=DataClassJSONEncoder,
            )


def check_folders_exist(*folders) -> bool:
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def extract_data(text_path: str, wav_folder: str) -> Dict[str, Data]:
    data: Dict[str, Data] = {}
    with open(text_path, "r", encoding="utf-8") as text_file:
        text_file_lines = text_file.readlines()
        for text_file_line in text_file_lines:
            split_line = text_file_line.split()
            data[split_line[0]] = Data(
                wav=os.path.join(wav_folder, f"{split_line[0]}.wav"),
                transcription=" ".join(split_line[1:]),
            )
    return data


def remove_useless_data(
    data: Dict[str, Data], keep_unk=False
) -> Dict[str, Data]:
    useful_data: Dict[str, Data] = {}

    check_useability_regex = r"[a-zA-Z]+"
    if keep_unk:
        for key, line in data.items():
            data[key] = Data(
                wav=line.wav,
                transcription=line.transcription.replace("UNK", "unk"),
            )
        check_useability_regex = r"[a-zA-Z]+\b(?<!\bunk)"

    for key, line in data.items():
        useless = bool(re.search(check_useability_regex, line.transcription))
        if not useless and len(line.transcription) > 0:
            useful_data[key] = line

    return useful_data


if __name__ == "__main__":
    save_folder = "results/prepare"
    dataset_folder = "PLACEHOLDER"
    prepare_matbn(dataset_folder, save_folder)
