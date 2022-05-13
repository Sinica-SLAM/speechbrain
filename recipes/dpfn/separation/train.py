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
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging


# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, mix, targets, targets_e, ids, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)
        targets_e = torch.cat(
            [
                targets_e[i][0].unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        ).to(self.device)
        ids = torch.cat(
            [
                torch.tensor(ids[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        ).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)

                    mix = targets.sum(-1)

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                if self.hparams.limit_training_signal_len:
                    mix, targets, targets_e = self.cut_signals(
                        mix, targets, targets_e
                    )

                if self.hparams.rand_choice:
                    mix, targets, targets_e, ids = self.rand_choice(
                        mix, targets, targets_e, ids
                    )

        # Separation
        Batch, Time, Spk = targets.shape

        X = mix.unsqueeze(1).repeat(1, Spk, 1)
        X = X.flatten(0, 1)
        X_est = targets_e.permute(0, 2, 1).contiguous()
        X_est = X_est.flatten(0, 1).unsqueeze(1)

        mix_w = self.hparams.Encoder(X)
        X_est_w = self.hparams.Encoder(X_est.squeeze(1))
        C, mid, spec = self.hparams.SpkNet(X_est, X_est_w)
        est_mask = self.hparams.MaskNet(mix_w, C)
        _, N, L = mix_w.shape
        if isinstance(est_mask, list):
            est_source = []
            for est_m in est_mask:
                sep_h = mix_w * est_m
                sep_h = (
                    sep_h.view(Batch, Spk, N, L)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )

                # Decoding
                est_s = torch.cat(
                    [
                        self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                        for i in range(self.hparams.num_spks)
                    ],
                    dim=-1,
                )
                est_source.append(est_s)
        else:
            sep_h = mix_w * est_mask
            sep_h = (
                sep_h.view(Batch, Spk, N, L).permute(1, 0, 2, 3).contiguous()
            )

            # Decoding
            est_source = torch.cat(
                [
                    self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                    for i in range(self.hparams.num_spks)
                ],
                dim=-1,
            )

        if (
            self.hparams.loss_kind == "l1"
            or self.hparams.loss_kind == "mse"
            or self.hparams.loss_kind == "snr"
        ):
            est_spec = self.hparams.AutoEncoder(mid)
            if self.hparams.SpkNet.pre_encoding == "encoder":
                est_spec = self.hparams.Decoder(est_spec)
                est_spec = est_spec.view(Batch, Spk, -1)
                est_spec = est_spec.permute(0, 2, 1).contiguous()
                spec = targets

                T_origin = targets.size(1)
                T_est = est_spec.size(1)
                if T_origin > T_est:
                    est_spec = F.pad(est_spec, (0, 0, 0, T_origin - T_est))
                else:
                    est_spec = est_spec[:, :T_origin, :]
            else:
                _, N, L = est_spec.shape
                est_spec = est_spec.view(Batch, Spk, N, L)
                spec = spec.view(Batch, Spk, N, L)
            pred_spks = 0
        elif self.hparams.loss_kind == "spk":
            pred_spks = self.hparams.SpkClassifier(C)
            pred_spks = pred_spks.view(Batch, Spk, -1)
            est_spec = 0
        else:
            raise NotImplementedError(
                f"We haven't implemented loss kind: {self.hparams.loss_kind}"
            )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        if isinstance(est_source, list):
            for i in range(len(est_source)):
                T_est = est_source[i].size(1)
                if T_origin > T_est:
                    est_source[i] = F.pad(
                        est_source[i], (0, 0, 0, T_origin - T_est)
                    )
                else:
                    est_source[i] = est_source[i][:, :T_origin, :]
        else:
            T_est = est_source.size(1)
            if T_origin > T_est:
                est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
            else:
                est_source = est_source[:, :T_origin, :]

        return est_source, targets, est_spec, spec, pred_spks, ids

    def compute_objectives(
        self,
        predictions,
        targets,
        pred_spec,
        spec,
        pred_spks,
        ids,
        weight,
        kind,
        stage,
    ):
        """Computes the sinr loss"""
        return self.hparams.loss(
            targets,
            predictions,
            pred_spec,
            spec,
            pred_spks,
            ids,
            weight,
            kind,
            stage,
        )

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        targets_e = [batch.s1_e_sig, batch.s2_e_sig]
        ids = [batch.s1_ID, batch.s2_ID]

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.hparams.auto_mix_prec:
            with autocast():
                (
                    predictions,
                    targets,
                    pred_spec,
                    spec,
                    pred_spks,
                    ids,
                ) = self.compute_forward(
                    mixture, targets, targets_e, ids, sb.Stage.TRAIN
                )
                loss_total = self.compute_objectives(
                    predictions,
                    targets,
                    pred_spec,
                    spec,
                    pred_spks,
                    ids,
                    self.hparams.loss_weight,
                    self.hparams.loss_kind,
                    "TRAIN",
                )

                if isinstance(loss_total, dict):
                    loss = (
                        loss_total["si_snr"]
                        + self.hparams.loss_weight * loss_total["spk_loss"]
                    )
                else:
                    loss = loss_total

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
            (
                predictions,
                targets,
                pred_spec,
                spec,
                pred_spks,
                ids,
            ) = self.compute_forward(
                mixture, targets, targets_e, ids, sb.Stage.TRAIN
            )
            loss_total = self.compute_objectives(
                predictions,
                targets,
                pred_spec,
                spec,
                pred_spks,
                ids,
                self.hparams.loss_weight,
                self.hparams.loss_kind,
                "TRAIN",
            )

            if isinstance(loss_total, dict):
                loss = (
                    loss_total["si_snr"]
                    + self.hparams.loss_weight * loss_total["spk_loss"]
                )

                for l_kind in loss_total.keys():
                    l_k = loss_total[l_kind]
                    loss_total[l_kind] = self.threshold_byloss(l_k)

            else:
                loss = loss_total
                loss_total = self.threshold_byloss(loss_total)

            loss = self.threshold_byloss(loss)

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computation

                # # normalize the loss by gradient_accumulation step
                # (loss / self.hparams.gradient_accumulation).backward()

                # if self.hparams.clip_grad_norm >= 0:
                #     torch.nn.utils.clip_grad_norm_(
                #         self.modules.parameters(), self.hparams.clip_grad_norm
                #     )

                # if self.step % self.hparams.gradient_accumulation == 0:
                #     # gradient clipping & early stop if loss is not fini
                #     self.check_gradients(loss)

                #     self.optimizer.step()

                #     # anneal lr every update
                #     self.hparams.noam_annealing(self.optimizer)

                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )
                if isinstance(
                    self.hparams.lr_scheduler, schedulers.NoamScheduler
                ):
                    current_lr, next_lr = self.hparams.lr_scheduler(
                        self.optimizer
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

        if isinstance(loss_total, dict):
            for key in loss_total.keys():
                loss_total[key] = loss_total[key].detach().cpu()
        else:
            loss_total = loss_total.detach().cpu()
        return loss_total

    def threshold_byloss(self, _loss):
        if self.hparams.threshold_byloss:
            th = self.hparams.threshold
            l_to_keep = _loss[_loss > th]
            if l_to_keep.nelement() > 0:
                l_total = l_to_keep.mean()
            else:
                l_total = torch.zeros(1, requires_grad=True).to(
                    l_to_keep.device
                )
        else:
            l_total = _loss.mean()
        return l_total

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        targets_e = [batch.s1_e_sig, batch.s2_e_sig]
        ids = [batch.s1_ID, batch.s2_ID]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            (
                predictions,
                targets,
                pred_spec,
                spec,
                pred_spks,
                ids,
            ) = self.compute_forward(mixture, targets, targets_e, ids, stage)
            loss = self.compute_objectives(
                predictions,
                targets,
                pred_spec,
                spec,
                pred_spks,
                ids,
                1,
                self.hparams.loss_kind,
                "TEST",
            )
            loss = loss.mean()
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
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = {"loss": stage_loss}

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
            elif isinstance(
                self.hparams.lr_scheduler, schedulers.NoamScheduler
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(self.optimizer)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def update_average(self, loss, avg_loss):
        """Update running average of the loss.
        Arguments
        ---------
        loss : torch.tensor
            detached loss, a single float value.
        avg_loss : float
            current running average.
        Returns
        -------
        avg_loss : float
            The average loss.
        """
        if isinstance(loss, dict):
            if avg_loss == 0:
                avg_loss = loss.copy()
                for key in loss.keys():
                    avg_loss[key] /= self.step
            else:
                for key in loss.keys():
                    if torch.isfinite(loss[key]):
                        avg_loss[key] -= avg_loss[key] / self.step
                        avg_loss[key] += float(loss[key]) / self.step
        else:
            if torch.isfinite(loss):
                avg_loss -= avg_loss / self.step
                avg_loss += float(loss) / self.step
        return avg_loss

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

    def cut_signals(self, mixture, targets, targets_e):
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
        targets_e = targets_e[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        return mixture, targets, targets_e

    def rand_choice(self, mix, targets, targets_e, ids):
        """
        This function randomly switches the s1 signal and s2 signal or estimated and source
        """
        if np.random.choice([True, False], p=[0.5, 0.5]):
            targets = torch.cat(
                [targets[..., 1].unsqueeze(-1), targets[..., 0].unsqueeze(-1)],
                dim=-1,
            )
            targets_e = torch.cat(
                [
                    targets_e[..., 1].unsqueeze(-1),
                    targets_e[..., 0].unsqueeze(-1),
                ],
                dim=-1,
            )
            ids = torch.cat(
                [ids[..., 1].unsqueeze(-1), ids[..., 0].unsqueeze(-1)], dim=-1,
            )

            if np.random.choice(
                [True, False],
                p=[self.hparams.rand_ori_rate, 1 - self.hparams.rand_ori_rate],
            ):
                targets_e = torch.cat(
                    [
                        targets[..., 0].unsqueeze(-1),
                        targets_e[..., 1].unsqueeze(-1),
                    ],
                    dim=-1,
                )

            if np.random.choice(
                [True, False],
                p=[self.hparams.rand_ori_rate, 1 - self.hparams.rand_ori_rate],
            ):
                targets_e = torch.cat(
                    [
                        targets_e[..., 0].unsqueeze(-1),
                        targets[..., 1].unsqueeze(-1),
                    ],
                    dim=-1,
                )

        return mix, targets, targets_e, ids

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
            test_data, **self.hparams.testloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    targets_e = [batch.s1_e_sig, batch.s2_e_sig]
                    ids = [batch.s1_ID, batch.s2_ID]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        (
                            predictions,
                            targets,
                            est_spec,
                            spec,
                            pred_spks,
                            ids,
                        ) = self.compute_forward(
                            batch.mix_sig,
                            targets,
                            targets_e,
                            ids,
                            sb.Stage.TEST,
                        )

                    if isinstance(predictions, list):
                        predictions = predictions[-1]

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(
                        predictions,
                        targets,
                        est_spec,
                        spec,
                        pred_spks,
                        ids,
                        1.0,
                        self.hparams.loss_kind,
                        "TEST",
                    )

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal,
                        targets,
                        est_spec,
                        spec,
                        pred_spks,
                        ids,
                        1.0,
                        self.hparams.loss_kind,
                        "TEST",
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
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

        if isinstance(predictions, list):
            predictions = predictions[-1]
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

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        return mix_sig

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = sb.dataio.dataio.read_audio(s2_wav)
        return s2_sig

    @sb.utils.data_pipeline.takes("s1_id")
    @sb.utils.data_pipeline.provides("s1_ID")
    def id_pipeline_s1(s1_id):
        s1_ID = int(s1_id)
        return s1_ID

    @sb.utils.data_pipeline.takes("s2_id")
    @sb.utils.data_pipeline.provides("s2_ID")
    def id_pipeline_s2(s2_id):
        s2_ID = int(s2_id)
        return s2_ID

    @sb.utils.data_pipeline.takes("s1_e_wav")
    @sb.utils.data_pipeline.provides("s1_e_sig")
    def audio_pipeline_s1_e(s1_e_wav):
        s1_e_sig = sb.dataio.dataio.read_audio(s1_e_wav)
        return s1_e_sig

    @sb.utils.data_pipeline.takes("s2_e_wav")
    @sb.utils.data_pipeline.provides("s2_e_sig")
    def audio_pipeline_s2_e(s2_e_wav):
        s2_e_sig = sb.dataio.dataio.read_audio(s2_e_wav)
        return s2_e_sig

    if hparams["num_spks"] == 3:

        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            s3_sig = sb.dataio.dataio.read_audio(s3_wav)
            return s3_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1_e)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2_e)
    sb.dataio.dataset.add_dynamic_item(datasets, id_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, id_pipeline_s2)
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets,
            [
                "id",
                "mix_sig",
                "s1_sig",
                "s2_sig",
                "s1_e_sig",
                "s2_e_sig",
                "s1_ID",
                "s2_ID",
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
    from recipes.dpfn.prepare_data import prepare_wsjmix  # noqa

    run_on_main(
        prepare_wsjmix,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
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
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)