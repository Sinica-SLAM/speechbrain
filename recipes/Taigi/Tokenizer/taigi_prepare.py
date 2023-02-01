import logging
import random
import os

from dataclasses import dataclass, is_dataclass, asdict
from typing import List, Dict

import json

TRAIN_SPLIT: int = 360000  # 100 hr
DEV_SPLIT: int = 72000  # 20 hr
TEST_SPLIT: int = 72000  # 20 hr

logger = logging.getLogger(__name__)


@dataclass
class Data:
    translation: str
    wav: str
    duration: float


class DataClassJSONEncoder(json.JSONEncoder):
    def default(self, object):
        if is_dataclass(object):
            return asdict(object)
        return super().default(object)


def prepare_taigi(
    dataset_folder: str, save_folder: str, seed: int, skip_prep: bool = False,
):
    wav_folder = os.path.join(dataset_folder, "wav")
    data_folder = os.path.join(dataset_folder, "data")

    if skip_prep:
        return

    if check_folders_exist(wav_folder, data_folder) is not True:
        logger.error(
            "the folder wav or data does not exist (it is expected in the "
            "Taigi dataset)"
        )

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    data_path = os.path.join(data_folder, "text_mandarin")

    data = extract_data(data_path=data_path)
    datasets = split_datasets(data=data, seed=seed)

    for index, split in enumerate(["train", "dev", "test"]):
        with open(
            f"{save_folder}/{split}.json", "w+", encoding="utf-8"
        ) as save_file:
            json.dump(
                datasets[index],
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


def extract_data(data_path: str) -> List[Dict[str, Data]]:
    extracted_data = []
    with open(data_path, "r", encoding="utf-8") as text_file:
        text_file_lines = text_file.readlines()
        for text_file_line in text_file_lines:
            text_file_line = text_file_line.split()
            wav_id = text_file_line[0]
            translation = " ".join(text_file_line[1:])
            durations = wav_id.split("_")[-1].split("-")
            start, end = int(durations[0]), int(durations[1])
            duration = (end - start) / 1000

            extracted_data.append(
                {
                    "wav_id": wav_id,
                    "data": Data(
                        translation=translation,
                        wav="{data_root}/wav/" + f"{wav_id}.wav",
                        duration=duration,
                    ),
                }
            )

    return extracted_data


def split_datasets(
    data: List[Dict[str, Data]], seed: int
) -> List[Dict[str, Data]]:
    random.Random(seed).shuffle(data)

    duration_sum = 0
    train, dev, test = {}, {}, {}
    for sample in data:
        duration_sum += sample["data"].duration
        if duration_sum < TRAIN_SPLIT:
            train[sample["wav_id"]] = sample["data"]
        elif duration_sum < TRAIN_SPLIT + DEV_SPLIT:
            dev[sample["wav_id"]] = sample["data"]
        elif duration_sum < TRAIN_SPLIT + DEV_SPLIT + TEST_SPLIT:
            test[sample["wav_id"]] = sample["data"]
        else:
            pass

    train_duration = sum(sample.duration for sample in train.values()) / 3600
    dev_duration = sum(sample.duration for sample in dev.values()) / 3600
    test_duration = sum(sample.duration for sample in test.values()) / 3600

    logging.info(
        f"train: {train_duration} dev: {dev_duration} test: {test_duration}"
    )

    return [train, dev, test]


if __name__ == "__main__":
    prepare_taigi(
        dataset_folder="/mnt/md0/user_jamfly/CORPUS/taigi",
        save_folder="/mnt/md0/user_jamfly/sb_data/taigi",
        seed=1234,
        skip_prep=False,
    )
