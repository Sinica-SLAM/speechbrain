#!/usr/bin/env/python3
import sys
import json
import subprocess

from typing import Dict

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from speechbrain.utils.distributed import run_on_main


def read_json(json_path: str) -> Dict[str, str]:
    """Read the given json file, and return a dictionary"""
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data


def read_train_file(
    train_text_path: str, data_json: Dict[str, str], phone_tokenizer=None,
):
    """Read the training text from the given json file"""
    with open(train_text_path, "w+", encoding="utf-8") as train_text_file:
        for value in data_json.values():
            if phone_tokenizer is not None:
                allosaurus_for_bpe = (
                    value["allosaurus_for_bpe"].strip().lstrip()
                )
                phone_sequence = phone_tokenizer.encode_as_pieces(
                    allosaurus_for_bpe
                )
                phone_sequence = " ".join(phone_sequence)
            else:
                phone_sequence = value["allosaurus"].strip().lstrip()
            train_text_file.write(phone_sequence + "\n")


def create_lexicon(train_text_path: str, lexicon_path: str):
    """Create the lexicon file from the training data"""
    with open(train_text_path, "r", encoding="utf-8") as trian_text_file, open(
        lexicon_path, "w+", encoding="utf-8"
    ) as lexicon_file:

        train_text_lines = trian_text_file.readlines()

        lexicon = []
        for train_text_line in train_text_lines:
            lexicon += train_text_line.strip().lstrip().split(" ")

        lexicon = set(lexicon)
        lexicon = sorted(lexicon)

        if "" in lexicon:
            lexicon.remove("")

        lexicon_file.write("\n".join(lexicon))


def train_lm(lexicon_path: str, train_text_path: str, lm_path: str):
    """Train the arpa language model"""
    command = (
        "ngram-count "
        f"-vocab {lexicon_path} "
        f"-order 3 -text {train_text_path} "
        "-interpolate -wbdiscount -wbdiscount1 -wbdiscount2 "
        f"-lm {lm_path}"
    ).split(" ")
    subprocess.run(command)


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    phone_tokenizer = None
    if "phone_pretrainer" in hparams:
        run_on_main(hparams["phone_pretrainer"].collect_files)
        hparams["phone_pretrainer"].load_collected(device=run_opts["device"])
        phone_tokenizer = hparams["phone_tokenizer"]

    data_json = read_json(json_path=hparams["data_json_path"])
    read_train_file(
        train_text_path=hparams["train_text_path"],
        data_json=data_json,
        phone_tokenizer=phone_tokenizer,
    )

    create_lexicon(
        train_text_path=hparams["train_text_path"],
        lexicon_path=hparams["lexicon_path"],
    )

    train_lm(
        lexicon_path=hparams["lexicon_path"],
        train_text_path=hparams["train_text_path"],
        lm_path=hparams["arpa_lm_path"],
    )
