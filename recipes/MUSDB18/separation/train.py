#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import torch

# import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import json
import logging

# from utils import TensorChunk, center_trim
import time


# Define training procedure
class Separation(sb.Brain):
    def __init__(  # noqa: C901
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        normalize_data=None,
    ):

        super(Separation, self).__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )

        self.normalize_data = normalize_data

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):

        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:

            # Training stage
            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    t.set_postfix(train_loss=self.avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        run_on_main(self._save_intra_epoch_ckpt)
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[sb.Stage.VALID, avg_valid_loss, epoch],
                    )

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break

    def normalize(self, names):
        with open(self.normalize_data) as file:
            f = json.load(file)

        norm = [f[name] for name in names]

        return norm

    def compute_forward(self, mix, targets, stage, songnames=None, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        # print(mix)
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        mix = mix.mean(dim=-1)
        # print('mix', mix.shape)

        # Convert targets to tensor, and calculate Mean from stereo to mono
        targets = torch.cat(
            [
                torch.mean(targets[i][0], dim=-1).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        ).to(self.device)

        # Normalize
        norm = self.normalize(songnames)
        mix = (mix - norm[0]["mean"]) / norm[0]["std"]
        targets = (targets - norm[0]["mean"]) / norm[0]["std"]

        # Add speech distortions
        # print(mix.shape)
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)
                    mix = targets.sum(-1)

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                # Random segments for trainning
                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        # print('est',est_source.shape, 'targets', targets.shape)
        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""

        # print(targets.shape, predictions.shape)
        loss = self.hparams.loss(targets, predictions)

        predictions = predictions.clone().permute(1, 0, 2)
        targets = targets.clone().permute(1, 0, 2)
        sisnr = self.hparams.sisnr(targets, predictions).mean()

        return loss, sisnr

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        songnames = [
            song.split("/")[-1].replace(".wav", "") for song in batch.mix_wav
        ]
        # print(songnames)

        mixture = batch.mix_sig
        targets = [
            batch.vocals_sig,
            batch.bass_sig,
            batch.drums_sig,
            batch.other_sig,
        ]

        if self.hparams.auto_mix_prec:
            with autocast():
                predictions, targets = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN, songnames,
                )
                loss, sisnr = self.compute_objectives(predictions, targets)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        else:
            predictions, targets = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN, songnames
            )
            loss, sisnr = self.compute_objectives(predictions, targets)

            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [
            batch.vocals_sig,
            batch.bass_sig,
            batch.drums_sig,
            batch.other_sig,
        ]

        songnames = [
            song.split("/")[-1].replace(".wav", "") for song in batch.mix_wav
        ]

        with torch.no_grad():
            predictions, targets = self.compute_forward(
                mixture, targets, stage, songnames
            )
            loss, sisnr = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        # Prepare lists to compute metrics for full tracks
        concat_mixture = []
        concat_predictions = []
        concat_targets = []
        song_id = 0

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [
                        batch.vocals_sig,
                        batch.bass_sig,
                        batch.drums_sig,
                        batch.other_sig,
                    ]
                    songnames = [
                        song.split("/")[-1].replace(".wav", "")
                        for song in batch.mix_wav
                    ]

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            (mixture, mix_len),
                            targets,
                            sb.Stage.TEST,
                            songnames,
                        )

                        norm = self.normalize(songnames)
                        predictions = (
                            predictions * norm[0]["std"] + norm[0]["mean"]
                        )
                        targets = targets * norm[0]["std"] + norm[0]["mean"]

                    if (
                        snt_id[0].split("_")[0] == str(song_id)
                        and i != len(t) - 1
                    ):
                        concat_mixture.append(mixture)
                        concat_predictions.append(predictions)
                        concat_targets.append(targets)

                    else:
                        concat_mixture = torch.cat(concat_mixture, dim=1)
                        concat_predictions = torch.cat(
                            concat_predictions, dim=1
                        )
                        concat_targets = torch.cat(concat_targets, dim=1)

                        # Compute SI-SNR
                        loss, sisnr = self.compute_objectives(
                            concat_predictions, concat_targets
                        )

                        # Compute SI-SNR improvement
                        mixture_signal = torch.stack(
                            [concat_mixture] * self.hparams.num_spks, dim=-1
                        )

                        # print(mixture_signal.shape)
                        mixture_signal = mixture_signal.to(targets.device)

                        # Stereo to mono
                        mixture_signal = torch.mean(mixture_signal, dim=2)

                        # print(mixture_signal.shape, concat_targets.shape)
                        # print(concat_targets[0].shape, concat_predictions[0].shape, mixture_signal[0].shape)

                        loss, sisnr_baseline = self.compute_objectives(
                            mixture_signal, concat_targets
                        )
                        sisnr_i = sisnr - sisnr_baseline

                        # Compute SDR
                        sdr, _, _, _ = bss_eval_sources(
                            concat_targets[0].t().cpu().numpy(),
                            concat_predictions[0].t().detach().cpu().numpy(),
                        )

                        sdr_baseline, _, _, _ = bss_eval_sources(
                            concat_targets[0].t().cpu().numpy(),
                            mixture_signal[0].t().detach().cpu().numpy(),
                        )

                        sdr_i = sdr.mean() - sdr_baseline.mean()

                        # Saving on a csv file
                        row = {
                            "snt_id": song_id,
                            "sdr": sdr.mean(),
                            "sdr_i": sdr_i,
                            "si-snr": -sisnr.item(),
                            "si-snr_i": -sisnr_i.item(),
                        }
                        writer.writerow(row)

                        # Metric Accumulation
                        all_sdrs.append(sdr.mean())
                        all_sdrs_i.append(sdr_i.mean())
                        all_sisnrs.append(-sisnr.item())
                        all_sisnrs_i.append(-sisnr_i.item())

                        # Next song
                        song_id += 1
                        concat_mixture = [mixture]
                        concat_predictions = [predictions]
                        concat_targets = [targets]

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines
    @sb.utils.data_pipeline.takes("mix_wav", "start", "stop")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav, start, stop):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        mix_sig, fs = torchaudio.load(
            mix_wav, num_frames=num_frames, frame_offset=start
        )
        mix_sig = mix_sig.transpose(0, 1)

        return mix_sig

    @sb.utils.data_pipeline.takes("vocals_wav", "start", "stop")
    @sb.utils.data_pipeline.provides("vocals_sig")
    def audio_pipeline_vocals(vocals_wav, start, stop):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        vocals_sig, fs = torchaudio.load(
            vocals_wav, num_frames=num_frames, frame_offset=start
        )
        vocals_sig = vocals_sig.transpose(0, 1)

        return vocals_sig

    @sb.utils.data_pipeline.takes("bass_wav", "start", "stop")
    @sb.utils.data_pipeline.provides("bass_sig")
    def audio_pipeline_bass(bass_wav, start, stop):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        bass_sig, fs = torchaudio.load(
            bass_wav, num_frames=num_frames, frame_offset=start
        )
        bass_sig = bass_sig.transpose(0, 1)

        return bass_sig

    @sb.utils.data_pipeline.takes("drums_wav", "start", "stop")
    @sb.utils.data_pipeline.provides("drums_sig")
    def audio_pipeline_drums(drums_wav, start, stop):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        drums_sig, fs = torchaudio.load(
            drums_wav, num_frames=num_frames, frame_offset=start
        )
        drums_sig = drums_sig.transpose(0, 1)

        return drums_sig

    @sb.utils.data_pipeline.takes("other_wav", "start", "stop")
    @sb.utils.data_pipeline.provides("other_sig")
    def audio_pipeline_other(other_wav, start, stop):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        other_sig, fs = torchaudio.load(
            other_wav, num_frames=num_frames, frame_offset=start
        )
        other_sig = other_sig.transpose(0, 1)

        return other_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_vocals)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_bass)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_drums)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_other)
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "mix_wav",
            "mix_sig",
            "vocals_sig",
            "bass_sig",
            "drums_sig",
            "other_sig",
        ],
    )

    return train_data, valid_data, test_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    run_opts["auto_mix_prec"] = hparams["auto_mix_prec"]

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        print(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )
        sys.exit(1)

    # Data preparation
    from recipes.MUSDB18.prepare_data import prepare_musdb18  # noqa

    run_on_main(
        prepare_musdb18,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "origin_datapath": hparams["origin_datapath"],
            "target_datapath": hparams["target_datapath"],
            "skip_prep": hparams["skip_prep"],
            "train_seg_dur": hparams["train_seg_dur"],
            "valid_seg_dur": hparams["valid_seg_dur"],
            "fs": hparams["sample_rate"],
        },
    )

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        from dynamic_mixing import dynamic_mix_data_prep

        # if the base_folder for dm is not processed, preprocess them
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from recipes.WSJ0Mix.meta.preprocess_dynamic_mixing import (
                    resample_folder,
                )

                print("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                        + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.wav",
                    },
                )
                # adjust the base_folder_dm path
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                print(
                    "Using the existing processed folder on the same directory as base_folder_dm"
                )
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        train_data = dynamic_mix_data_prep(hparams)
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        normalize_data=hparams["nomalize_data"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    separator.evaluate(test_data, min_key="loss")
    # separator.device = 'cpu'
    separator.save_results(test_data)
