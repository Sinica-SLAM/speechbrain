import sys

import torch
import speechbrain as sb
from speechbrain.dataio import dataset
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from train_time_rnn_lm import LM
import glob
import os
import pandas as pd
import datetime


def dataio_prepare(hparams):
    @sb.utils.data_pipeline.takes("transcription")
    @sb.utils.data_pipeline.provides(
        "transcription", "tokens_bos", "tokens_eos"
    )
    def transcription_pipline(transcription):
        yield transcription
        tokens_list = hparams["tokenizer"].encode_as_ids(transcription)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos

    @sb.utils.data_pipeline.takes("date")
    @sb.utils.data_pipeline.provides("date_token")
    def date_pipline(date):
        date_token = hparams["date_tokenizer"].encode(
            datetime.datetime.strptime(date, "%Y%m%d").date()
        )
        yield date_token

    data_folder = "results/prepare_cna_week"
    datasets = {}
    for json_file_path in glob.glob(os.path.join(data_folder, "*.json")):
        json_file_name = os.path.basename(json_file_path).replace(".json", "")
        datasets[json_file_name] = dataset.DynamicItemDataset.from_json(
            json_path=json_file_path,
            replacements={"data_root": data_folder},
            dynamic_items=[transcription_pipline, date_pipline],
            output_keys=[
                "transcription",
                "tokens_bos",
                "tokens_eos",
                "date_token",
            ],
        )

    return datasets


if __name__ == "__main__":
    hparams_file_path, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file_path) as hparams_file:
        hparams = load_hyperpyyaml(hparams_file, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file_path,
        overrides=overrides,
    )

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    datasets = dataio_prepare(hparams)

    lm_brain = LM(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    weeks = []
    loss = []

    for week_name in sorted(datasets.keys()):

        week_loss = lm_brain.evaluate(
            datasets[week_name],
            min_key="loss",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
        weeks.append(week_name)
        loss.append(week_loss)
    pd.DataFrame({"week": weeks, "loss": loss}).to_csv(
        os.path.join(hparams["output_folder"], "week_loss.csv"), index=False
    )
