import logging
import os
from dataclasses import dataclass, is_dataclass, asdict

import json

logger = logging.getLogger(__name__)


@dataclass
class Data:
    date: str
    transcription: str


class DataClassJSONEncoder(json.JSONEncoder):
    def default(self, object):
        if is_dataclass(object):
            return asdict(object)
        return super().default(object)


def prepare_cna(
    dataset_folder: str,
    save_folder: str,
    settings_json_path: str,
    before_2000: bool,
    skip_prep: bool = False,
):
    if skip_prep:
        return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if check_folders_exist(dataset_folder) is not True:
        logger.error("the dataset folder does not exist)")

    settings = {"train": [], "test": [], "valid": []}

    # load setting
    with open(settings_json_path, "r") as settings_file:
        settings = json.load(settings_file)

    for split_name in ["valid", "test", "train"]:
        data = {}
        for text_file_name in settings[split_name]:
            if not before_2000 and not text_file_name.startswith("20"):
                continue

            text_file_path = os.path.join(
                dataset_folder, f"{text_file_name}.txt"
            )
            with open(text_file_path, "r", encoding="utf-8") as text_file:
                for line in text_file.read().splitlines():
                    if len(line) > 128 or len(line) < 1:
                        continue
                    data[len(data)] = Data(
                        date=text_file_name, transcription=line
                    )
                text_file.close()

        save_path = os.path.join(save_folder, f"{split_name}.json")
        with open(save_path, "w", encoding="utf-8") as save_file:
            json.dump(
                data,
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


if __name__ == "__main__":
    save_folder = "results/prepare_cna"
    dataset_folder = "PLACEHOLDER"
    settings_json_path = "settings.json"
    before_2000 = False
    prepare_cna(dataset_folder, save_folder, settings_json_path, before_2000)
