import os
import json
import subprocess

from typing import Dict, Any

from tqdm import tqdm


def read_json(json_path: str) -> Dict[str, str]:
    """Read the given json file, and return a dictionary"""
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data


def save_json(json_path: str, data: Dict[str, Any]):
    """Save the given data as json"""
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    json_path = "/mnt/md0/user_jamfly/sb_data/Fisher_data_mid/train/data.json"
    wav_folder = "/mnt/md0/user_jamfly/sb_data/Fisher_data_mid/train-sp/wav"
    data_root = "/mnt/md0/user_jamfly/sb_data/Fisher_data_mid"

    data_json = read_json(json_path=json_path)

    if not os.path.isdir(wav_folder):
        os.makedirs(wav_folder)

    sp_data_json = {}
    for sp in tqdm([0.9, 1.0, 1.1]):
        for utt_id, properties in tqdm(data_json.items()):
            sp_data = {}
            for property, value in properties.items():
                sp_data[property] = value

            if sp != 1.0:
                original_path = properties["wav"].replace(
                    "{data_root}", data_root
                )
                sp_wav_path = f"{wav_folder}/{sp}-{utt_id}.wav"

                subprocess.run(
                    f"sox -G {original_path} {sp_wav_path} speed {sp}".split()
                )

                wav_path = "{data_root}" + f"/train-sp/wav/{sp}-{utt_id}.wav"
            else:
                wav_path = properties["wav"]

            sp_data["wav"] = wav_path
            sp_data["duration"] = round(int(properties["duration"]) / sp, 2)
            uid = f"{sp}-{utt_id}"
            sp_data_json[uid] = sp_data

    save_json(
        json_path="/mnt/md0/user_jamfly/sb_data/Fisher_data_mid/train-sp/data.json",
        data=sp_data_json,
    )
