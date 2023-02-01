import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

if __name__ == "__main__":
    hparams_file_path, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file_path) as hparams_file:
        hparams = load_hyperpyyaml(hparams_file, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file_path,
        overrides=overrides,
    )

    from taigi_prepare import prepare_taigi

    run_on_main(
        prepare_taigi,
        kwargs={
            "dataset_folder": hparams["dataset_folder"],
            "save_folder": hparams["save_folder"],
            "seed": hparams["seed"],
        },
    )

    hparams["tokenizer"]()
