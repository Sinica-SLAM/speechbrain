#!/usr/bin/python3
"""Recipe for training a speaker verification system based on PLDA using the voxceleb dataset.
The system employs a pre-trained model followed by a PLDA transformation.
The pre-trained model is automatically downloaded from the web if not specified.

To run this recipe, run the following command:
    >  python speaker_verification_plda.py hyperparams/verification_plda_xvector.yaml

Authors
    * Nauman Dawalatabad 2020
    * Mirco Ravanelli 2020
"""

import os
import sys
import torch
import torchaudio
import logging
import speechbrain as sb
import numpy
import pickle
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing.PLDA_LDA import Ndx
from speechbrain.processing.PLDA_LDA import fast_PLDA_scoring
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main


# Compute embeddings from the waveforms
def compute_embeddings(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    wavs = wavs.to(params["device"])
    wav_lens = wav_lens.to(params["device"])
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
        embeddings = params["mean_var_norm_emb"](
            embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device)
        )
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader, split, output_folder):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    save_dir = os.path.join(output_folder, f'example_{split}')
    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            emb = compute_embeddings(wavs, lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
                [ex_id, utt_id] = seg_id.split('--')
                save_path = os.path.join(save_dir, ex_id)
                # save_path = os.path.join(save_dir, 'aux')
                os.makedirs(save_path, exist_ok=True)
                numpy.save(save_path + f'/{utt_id}', emb[i].detach().cpu())
                # numpy.save(save_path + f'/{seg_id}', emb[i].detach().cpu())
    return embedding_dict


def verification_performance(scores_plda):
    """Computes the Equal Error Rate give the PLDA scores"""

    # Create ids, labels, and scoring list for EER evaluation
    ids = []
    labels = []
    positive_scores = []
    negative_scores = []
    for line in open(veri_file_path):
        lab = int(line.split(" ")[0].rstrip().split(".")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip()

        # Assuming enrol_id and test_id are unique
        i = int(numpy.where(scores_plda.modelset == enrol_id)[0][0])
        j = int(numpy.where(scores_plda.segset == test_id)[0][0])

        s = float(scores_plda.scoremat[i, j])
        labels.append(lab)
        ids.append(enrol_id + "<>" + test_id)
        if lab == 1:
            positive_scores.append(s)
        else:
            negative_scores.append(s)

    # Clean variable
    del scores_plda

    # Final EER computation
    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    min_dcf, th = minDCF(
        torch.tensor(positive_scores), torch.tensor(negative_scores)
    )
    return eer, min_dcf


# Function to get mod and seg
def get_utt_ids_for_test(ids, data_dict):
    mod = [data_dict[x]["wav1"]["data"] for x in ids]
    seg = [data_dict[x]["wav2"]["data"] for x in ids]

    return mod, seg


def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    data_folder = params["data_folder"]

    # 1. Declarations:

    # Train data (used for normalization)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["train_data"], replacements={"data_root": data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration", select_n=params["n_train_snts"]
    )

    # Enrol data
    dev_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["dev_data"], replacements={"data_root": data_folder},
    )
    dev_data = dev_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, dev_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id"])

    # 4 Create dataloaders
    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **params["train_dataloader_opts"]
    )
    dev_dataloader = sb.dataio.dataloader.make_dataloader(
        dev_data, **params["dev_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == "__main__":

    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    download_file(params["verification_file"], veri_file_path)

    from wsj0_prepare import prepare_extr, prepare_dpfn  # noqa E402

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    logger.info("Data preparation")
    prepare_dpfn(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        splits=["train", "dev", "test"],
        split_ratio=[90, 10],
        seg_dur=3,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_dataloader, dev_dataloader, test_dataloader = dataio_prep(params)

    # # Initialize PLDA vars
    # modelset, segset = [], []
    # embeddings = numpy.empty(shape=[0, params["emb_dim"]], dtype=numpy.float64)

    # # Embedding file for train data
    # xv_file = os.path.join(
    #     params["save_folder"], "VoxCeleb1_train_embeddings_stat_obj.pkl"
    # )

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected()

    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])


    # Compute enrol and Test embeddings
    train_obj = compute_embedding_loop(train_dataloader, 'tr', params["save_folder"])
    dev_obj = compute_embedding_loop(dev_dataloader, 'cv', params["save_folder"])
    test_obj = compute_embedding_loop(test_dataloader, 'tt', params["save_folder"])


    # Cleaning variable
    del train_dataloader
    del dev_dataloader
    del test_dataloader
    del train_obj
    del dev_obj
    del test_obj
