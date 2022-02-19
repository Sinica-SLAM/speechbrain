import json
import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from matbn_prepare import prepare_matbn
from cna_prepare import prepare_cna

if __name__ == "__main__":
    hparams_file_path, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file_path) as hparams_file:
        hparams = load_hyperpyyaml(hparams_file, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file_path,
        overrides=overrides,
    )

    run_on_main(
        prepare_matbn,
        kwargs={
            "dataset_folder": hparams["dataset_folder"],
            "save_folder": hparams["prepare_folder"],
            "keep_unk": hparams["keep_unk"],
            "skip_prep": hparams["skip_prepare"],
        },
    )

    if hparams["cna"]:
        run_on_main(
            prepare_cna,
            kwargs={
                "dataset_folder": hparams["cna_dataset_folder"],
                "save_folder": hparams["cna_prepare_folder"],
                "settings_json_path": hparams["cna_settings_json_path"],
                "before_2000": hparams["cna_before_2000"],
                "skip_prep": hparams["cna_skip_prepare"],
            },
        )

        with open(
            hparams["cna_train_json"], encoding="utf-8"
        ) as cna_train_file, open(
            hparams["train_json"], encoding="utf-8"
        ) as train_json:
            cna_train_data = json.load(cna_train_file)
            train_data = json.load(train_json)

            train_data.update(cna_train_data)
            json.dump(
                train_data,
                open(hparams["all_train_json"], "w", encoding="utf-8"),
                indent=2,
                ensure_ascii=False,
            )

    hparams["tokenizer"]()
