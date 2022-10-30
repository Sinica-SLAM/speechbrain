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
import csv
import logging
import time
import random
import itertools
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import LoopedLoader
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from enum import Enum, auto


# Define training procedure
class Separation(sb.Brain):
    def __init__(self, 
            modules=None,
            opt_class=None,
            hparams=None,
            run_opts=None,
            checkpointer=None,
            writer=None,
        ):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer,)
        
        if writer is not None:
            self.writer = SummaryWriter(writer)
        else:
            log_dir = os.path.join(hparams["output_folder"], 'log')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            
        self.channel_order = {'android': 0, 'condenser':1, 'ios':2, 'lavalier':3, 'XYH-6-X':4, 'XYH-6-Y':5}
    
    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        test_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
        embed_loader_kwargs={},
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
        if test_set is not None and not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_set = self.make_dataloader(
                test_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        embed_set = self.make_dataloader(
            train_set, stage=sb.Stage.TRAIN, **embed_loader_kwargs
        )
        
        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:

            # Training stage
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
                    sep_loss, mix_loss, ori_sep_loss, chan_id_loss = self.fit_batch(batch, epoch)
                    self.avg_train_loss = self.update_average(
                        sep_loss, self.avg_train_loss
                    )
                    self.writer.add_scalar("train_loss(si_snr)", sep_loss, self.step + (epoch-1) * len(t))
                    self.writer.add_scalar("train_loss(si_snr w/ orig channel)", ori_sep_loss, self.step + (epoch-1) * len(t))
                    self.writer.add_scalar("train_loss(channel)", mix_loss, self.step + (epoch-1) * len(t))
                    self.writer.add_scalar("train_loss(channel id)", chan_id_loss, self.step + (epoch-1) * len(t))
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
                        
                    if self.step == len(train_set) // 2 and self.hparams.train_embed:
                        step_embed = 0
                        avg_embed_loss = 0.0
                        with tqdm(
                            embed_set,
                            initial=0,
                            dynamic_ncols=True,
                            disable=not enable,
                        ) as k:
                            for batch_embed in k:
                                step_embed += 1
                                embed_loss = self.fit_batch_embed(batch_embed, epoch)
                                avg_embed_loss = self.update_average(
                                    embed_loss, avg_embed_loss
                                )
                                self.writer.add_scalar("embed_loss", embed_loss, step_embed + (epoch-1) * len(k))
                                k.set_postfix(train_loss=avg_embed_loss)

            # Run train "on_stage_end" on all processes
            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                avg_test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=sb.Stage.VALID, epoch=epoch)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break
                    
                    avg_valid_loss = {'valid_loss':avg_valid_loss}
                    
                    if self.hparams.valid_test:
                        self.step = 0
                        for batch in tqdm(
                            test_set, dynamic_ncols=True, disable=not enable
                        ):
                            self.step += 1
                            loss = self.evaluate_batch(batch, stage=sb.Stage.VALID, epoch=epoch)
                            avg_test_loss = self.update_average(
                                loss, avg_test_loss
                            )
                        avg_valid_loss['test_loss'] = avg_test_loss
                        self.writer.add_scalar("valid_test_loss", avg_valid_loss['test_loss'], epoch)
                    
                    # Only run validation "on_stage_end" on main process
                    self.writer.add_scalar("valid_loss", avg_valid_loss['valid_loss'], epoch)
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[sb.Stage.VALID, avg_valid_loss, epoch],
                    )

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break
    
    def compute_forward(self, channel_mix, channel_targets, stage, pretrain=False, noise=None):
        """Forward computations from the mixture to the separated signals."""
        est_channel_source = []
        est_channel_mix = []
        new_channel_targets = []
        new_channel_ori_targets = []
        new_channel_mix = []
        channel_preds = []
        channel_ids = []
        if stage == sb.Stage.TRAIN:
            if self.hparams.use_toy:
                sep_channel = 0 ### original channel
            else:
                sep_channel = random.choice(range(len(channel_mix)))
            sep_mix = [channel_mix[sep_channel]]
            sep_targets = [channel_targets[sep_channel]]
        else:
            sep_mix = channel_mix
            sep_targets = channel_targets
        for i in range(len(sep_mix)):
            mix = sep_mix[i]
            # Unpack lists and put tensors in the right device
            mix, mix_lens = mix
            mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)
            
            # target channel code
            if self.hparams.channel_adapt and stage == sb.Stage.TRAIN:
                if self.hparams.use_toy:
                    targ_channel = 1 ## target channel
                elif not self.hparams.change_channel:
                    targ_channel = sep_channel
                else:
                    rand_channel = list(range(len(channel_mix)))
                    rand_channel.remove(sep_channel)
                    targ_channel = random.choice(range(len(rand_channel)))
                chan_id = torch.tensor([targ_channel] * mix.shape[0]).to(self.device)
                targ_mix = channel_mix[targ_channel]
                targ_mix, targ_mix_lens = targ_mix
                targ_mix, targ_mix_lens = targ_mix.to(self.device), targ_mix_lens.to(self.device)
            else:
                targ_mix = mix
                chan_id = torch.tensor([i] * mix.shape[0]).to(self.device)
                
            # if self.hparams.channel_adapt and stage == sb.Stage.TRAIN:
            #     targets = channel_targets[targ_channel]
            # else:
            #     targets = sep_targets[i]
            targets = sep_targets[i]
            
            ori_targets = sep_targets[i]
            ori_targets = torch.cat(
                [ori_targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
                dim=-1,
            ).to(self.device)
            
            # Convert targets to tensor
            targets = torch.cat(
                [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
                dim=-1,
            ).to(self.device)

            # Add speech distortions
            if stage == sb.Stage.TRAIN:
                with torch.no_grad():
                    if self.hparams.limit_training_signal_len:
                        mix, targets, ori_targets, targ_mix = self.cut_signals(mix, targets, ori_targets, targ_mix)
                    
                    if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                        targets, ori_targets, mix, targ_mix = self.add_speed_perturb(targets, ori_targets, mix, targ_mix, mix_lens)

                    if self.hparams.use_wavedrop:
                        mix = self.hparams.wavedrop(mix, mix_lens)

            # Separation
            mix_w = self.hparams.Encoder(mix)
            if self.hparams.channel_adapt:
                if self.hparams.use_toy:
                    batch_index = [0,1] ### change utterance
                elif self.hparams.change_utt_order:
                    batch_index = torch.randperm(targ_mix.shape[0])
                else:
                    batch_index = list(range(targ_mix.shape[0]))
                chan_mix = targ_mix[batch_index]
                
                if self.hparams.random_channel:
                    C_mean = torch.zeros_like(chan_mix).to(self.device)
                    C_std = C_mean + 1
                    chan_mix = torch.normal(mean=C_mean, std=C_std).to(self.device)
                elif self.hparams.train_one:
                    chan_mix = torch.ones_like(chan_mix).to(self.device)
                
                chan_mix_w = self.hparams.Encoder(chan_mix)
                C_channel = self.hparams.ChannelEncoder(chan_mix, chan_mix_w)
                channel_pred = self.hparams.ChannelClassifier(C_channel)
                
                if self.hparams.infer_0 and stage != sb.Stage.TRAIN:
                    x, gap = self.hparams.pre_MaskNet(mix_w, C=None)
                else:
                    x, gap = self.hparams.pre_MaskNet(mix_w, C=C_channel)
                if pretrain:
                    with torch.no_grad():
                        est_mask = self.hparams.post_MaskNet(x, gap)
                else:
                    est_mask = self.hparams.post_MaskNet(x, gap)
                est_mix_h = self.hparams.post_MaskNet_channel(x, gap)
                est_mix_h = est_mix_h.squeeze(0)
                est_mix_h = est_mix_h * mix_w
                est_mix = self.hparams.Decoder_channel(est_mix_h)
            else:
                x, gap = self.hparams.pre_MaskNet(mix_w, C=None)
                est_mask = self.hparams.post_MaskNet(x, gap)
                est_mix = torch.zeros_like(targ_mix).to(self.device)
                channel_pred = torch.zeros(targ_mix.shape[0], 5).to(self.device)
            
            if pretrain:
                with torch.no_grad():
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
            else:
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
                
            T_origin = targ_mix.size(1)
            T_est = est_mix.size(1)
            if T_origin > T_est:
                est_mix = F.pad(est_mix, (0, T_origin - T_est))
            else:
                est_mix = est_mix[:, :T_origin]
            
            est_channel_source.append(est_source)
            new_channel_targets.append(targets)
            new_channel_ori_targets.append(ori_targets)
            est_channel_mix.append(est_mix)
            new_channel_mix.append(targ_mix)
            channel_preds.append(channel_pred)
            channel_ids.append(chan_id)

        return est_channel_source, new_channel_targets, new_channel_ori_targets, est_channel_mix, new_channel_mix, channel_preds, channel_ids

    def compute_forward_embed(self, mixture, stage):
        """Forward computations from the mixture to the separated signals."""
        emb_mix = []
        for i in range(len(mixture)):
            mix = mixture[i]
            # Unpack lists and put tensors in the right device
            mix, mix_lens = mix
            mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)
            
            # Separation
            mix_w = self.hparams.Encoder(mix)
            C_mix = self.hparams.ChannelEncoder(mix, mix_w)
            
            emb_mix.append(C_mix)

        return emb_mix

    def compute_objectives(self, predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids):
        """Computes the sinr loss"""
        return self.hparams.loss(targets, predictions, ori_targets, pred_mix, mix, chan_preds, chan_ids, self.hparams.loss_kind, self.hparams.channel_adapt)
    
    def compute_objectives_embed(self, embed):
        """Computes the sinr loss"""
        return self.hparams.loss_embed(embed)

    def fit_batch(self, batch, epoch):
        """Trains one batch"""
        # Unpacking batch list
        mixtures = [
            batch.android_mix_sig, 
            batch.condenser_mix_sig, 
            batch.ios_mix_sig,
            batch.lavalier_mix_sig,
            batch.XYH6X_mix_sig,
            batch.XYH6Y_mix_sig,
        ]
        targets = [
            [batch.android_s1_sig, batch.android_s2_sig],
            [batch.condenser_s1_sig, batch.condenser_s2_sig],
            [batch.ios_s1_sig, batch.ios_s2_sig],
            [batch.lavalier_s1_sig, batch.lavalier_s2_sig],
            [batch.XYH6X_s1_sig, batch.XYH6Y_s2_sig],
            [batch.XYH6Y_s1_sig, batch.XYH6Y_s2_sig],
        ]

        if self.hparams.num_spks == 3:
            targets[0].append(batch.android_s3_sig)
            targets[1].append(batch.condenser_s3_sig)
            targets[2].append(batch.ios_s3_sig)
            targets[3].append(batch.lavalier_s3_sig)
            targets[4].append(batch.XYH6X_s3_sig)
            targets[5].append(batch.XYH6Y_s3_sig)

        if not self.hparams.use_test_channel:
            mixtures.pop(self.channel_order[self.hparams.test_channel])
            targets.pop(self.channel_order[self.hparams.test_channel])
        
        if self.hparams.use_toy:
            mixtures = [mixtures[0], mixtures[1]]
            targets = [targets[0], targets[1]]
        
        if epoch <= self.hparams.pretrain:
            pretrain = True
        else:
            pretrain = False
        if self.hparams.auto_mix_prec:
            with autocast():
                predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids = self.compute_forward(
                    mixtures, targets, sb.Stage.TRAIN, pretrain=pretrain
                )
                sep_loss, mix_loss, ori_sep_loss, chan_id_loss = self.compute_objectives(
                    predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids
                )

                loss = []
                for i in range(len(sep_loss)):
                    if epoch <= self.hparams.pretrain:
                        _loss = mix_loss[i]
                    else:
                        _loss = sep_loss[i]
                        if self.hparams.channel_adapt:
                            _loss += mix_loss[i] * self.hparams.loss_weight + chan_id_loss[i] * self.hparams.chan_id_weight
                    loss.append(_loss)
                for i in range(len(loss)):
                    # hard threshold the easy dataitems
                    if self.hparams.threshold_byloss:
                        th = self.hparams.threshold
                        loss_to_keep = loss[i][loss[i] > th]
                        if loss_to_keep.nelement() > 0:
                            loss[i] = loss_to_keep.mean()
                    else:
                        loss[i] = loss[i].mean()
                loss = sum(loss) / len(loss)
                if loss.dim() > 0:
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
            predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids = self.compute_forward(
                mixtures, targets, sb.Stage.TRAIN, pretrain=pretrain
            )
            sep_loss, mix_loss, ori_sep_loss, chan_id_loss = self.compute_objectives(
                predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids
            )

            loss = []
            for i in range(len(sep_loss)):
                if epoch <= self.hparams.pretrain:
                    _loss = mix_loss[i]
                else:
                    _loss = sep_loss[i]
                    if self.hparams.channel_adapt:
                        _loss += mix_loss[i] * self.hparams.loss_weight + chan_id_loss[i] * self.hparams.chan_id_weight
                loss.append(_loss)
            for i in range(len(loss)):
                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[i][loss[i] > th]
                    if loss_to_keep.nelement() > 0:
                        loss[i] = loss_to_keep.mean()
                else:
                    loss[i] = loss[i].mean()
            loss = sum(loss) / len(loss)
            if loss.dim() > 0:
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

        for i in range(len(sep_loss)):
            sep_loss[i] = sep_loss[i].mean().detach().cpu()
            mix_loss[i] = mix_loss[i].mean().detach().cpu()
            ori_sep_loss[i] = ori_sep_loss[i].mean().detach().cpu()
            chan_id_loss[i] = chan_id_loss[i].mean().detach().cpu()
            
        sep_loss = sum(sep_loss) / len(sep_loss)
        mix_loss = sum(mix_loss) / len(mix_loss)
        ori_sep_loss = sum(ori_sep_loss) / len(ori_sep_loss)
        chan_id_loss = sum(chan_id_loss) / len(chan_id_loss)
        
        return sep_loss, mix_loss, ori_sep_loss, chan_id_loss

    def evaluate_batch(self, batch, stage, epoch=1000):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixtures = [
            batch.android_mix_sig, 
            batch.condenser_mix_sig, 
            batch.ios_mix_sig,
            batch.lavalier_mix_sig,
            batch.XYH6X_mix_sig,
            batch.XYH6Y_mix_sig,
        ]
        targets = [
            [batch.android_s1_sig, batch.android_s2_sig],
            [batch.condenser_s1_sig, batch.condenser_s2_sig],
            [batch.ios_s1_sig, batch.ios_s2_sig],
            [batch.lavalier_s1_sig, batch.lavalier_s2_sig],
            [batch.XYH6X_s1_sig, batch.XYH6Y_s2_sig],
            [batch.XYH6Y_s1_sig, batch.XYH6Y_s2_sig],
        ]

        if self.hparams.num_spks == 3:
            targets[0].append(batch.android_s3_sig)
            targets[1].append(batch.condenser_s3_sig)
            targets[2].append(batch.ios_s3_sig)
            targets[3].append(batch.lavalier_s3_sig)
            targets[4].append(batch.XYH6X_s3_sig)
            targets[5].append(batch.XYH6Y_s3_sig)

        if stage == sb.Stage.VALID:
            if not self.hparams.use_test_channel:
                mixtures.pop(self.channel_order[self.hparams.test_channel])
                targets.pop(self.channel_order[self.hparams.test_channel])
        elif stage == sb.Stage.TEST:
            mixtures = [mixtures[self.channel_order[self.hparams.test_channel]]]
            targets = [targets[self.channel_order[self.hparams.test_channel]]]
        else:
            raise ValueError(f"There is no such stage as {stage} in evaluation")
            
        with torch.no_grad():
            predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids = self.compute_forward(mixtures, targets, stage)
            if epoch <= self.hparams.pretrain:
                _, loss, _, _ = self.compute_objectives(
                    predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids
                )
            else:
                loss, _, _, _ = self.compute_objectives(
                    predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids
                    )
            for i in range(len(loss)):
                loss[i] = loss[i].mean()
            loss = sum(loss) / len(loss)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixtures[0], targets[0], predictions[0], self.hparams.test_channel)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixtures[0], targets[0], predictions[0], self.hparams.test_channel)

        return loss.detach()

    def fit_batch_embed(self, batch, epoch):
        """Trains one batch"""
        # Unpacking batch list
        mixtures = [
            batch.android_mix_sig, 
            batch.condenser_mix_sig, 
            batch.ios_mix_sig,
            batch.lavalier_mix_sig,
            batch.XYH6X_mix_sig,
            batch.XYH6Y_mix_sig,
        ]

        if not self.hparams.use_test_channel:
            mixtures.pop(self.channel_order[self.hparams.test_channel])
        
        if self.hparams.auto_mix_prec:
            with autocast():
                embeds = self.compute_forward_embed(
                    mixtures, sb.Stage.TRAIN
                )
                embed_loss = self.compute_objectives_embed(embeds)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = embed_loss[embed_loss > th]
                    if loss_to_keep.nelement() > 0:
                        embed_loss = loss_to_keep.mean()
                else:
                    embed_loss = embed_loss.mean()

            if (
                embed_loss < self.hparams.loss_upper_lim and embed_loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(embed_loss).backward()
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
                embed_loss.data = torch.tensor(0).to(self.device)
        else:
            embeds = self.compute_forward_embed(
                mixtures, sb.Stage.TRAIN
            )
            embed_loss = self.compute_objectives_embed(embeds)

            # hard threshold the easy dataitems
            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = embed_loss[embed_loss > th]
                if loss_to_keep.nelement() > 0:
                    embed_loss = loss_to_keep.mean()
            else:
                embed_loss = embed_loss.mean()

            if (
                embed_loss < self.hparams.loss_upper_lim and embed_loss.nelement() > 0
            ):  # the fix for computational problems
                embed_loss.backward()
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
                embed_loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        embed_loss = embed_loss.mean().detach().cpu()            
        
        return embed_loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        if isinstance(stage_loss, dict):
            if self.hparams.save_valid_test == 'test' and self.hparams.valid_test:
                stage_loss = stage_loss['test_loss']
            else:
                stage_loss = stage_loss['valid_loss']
        stage_stats = {"si-snr": stage_loss}
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
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, ori_targets, mix, targ_mix, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            new_ori_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                if i == 0:
                    new_target = self.hparams.speedperturb(
                        targets[:, :, i], targ_lens
                    )
                else:
                    new_target = self.hparams.speedperturb.speed_perturb.resamplers[self.hparams.speedperturb.speed_perturb.samp_index](
                        targets[:, :, i]
                    )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]
            
            for i in range(ori_targets.shape[-1]):
                new_ori_target = self.hparams.speedperturb.speed_perturb.resamplers[self.hparams.speedperturb.speed_perturb.samp_index](
                    ori_targets[:, :, i]
                )
                new_ori_targets.append(new_ori_target)
                
            mix = self.hparams.speedperturb.speed_perturb.resamplers[self.hparams.speedperturb.speed_perturb.samp_index](mix)
            targ_mix = self.hparams.speedperturb.speed_perturb.resamplers[self.hparams.speedperturb.speed_perturb.samp_index](targ_mix)

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
                        new_target.shape[-1],
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                    ori_targets = torch.zeros(
                        ori_targets.shape[0],
                        new_ori_target.shape[-1],
                        ori_targets.shape[-1],
                        device=ori_targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:new_target.shape[-1]]
                for i, new_ori_target in enumerate(new_ori_targets):
                    ori_targets[:, :, i] = new_ori_targets[i][:, 0:new_ori_target.shape[-1]]

        return targets, ori_targets, mix, targ_mix

    def cut_signals(self, mixture, targets, ori_targets, targ_mix):
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
        ori_targets = ori_targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        targ_mix = targ_mix[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets, ori_targets, targ_mix

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
                    mixtures = [
                        batch.android_mix_sig, 
                        batch.condenser_mix_sig, 
                        batch.ios_mix_sig,
                        batch.lavalier_mix_sig,
                        batch.XYH6X_mix_sig,
                        batch.XYH6Y_mix_sig,
                    ]
                    targets = [
                        [batch.android_s1_sig, batch.android_s2_sig],
                        [batch.condenser_s1_sig, batch.condenser_s2_sig],
                        [batch.ios_s1_sig, batch.ios_s2_sig],
                        [batch.lavalier_s1_sig, batch.lavalier_s2_sig],
                        [batch.XYH6X_s1_sig, batch.XYH6Y_s2_sig],
                        [batch.XYH6Y_s1_sig, batch.XYH6Y_s2_sig],
                    ]

                    if self.hparams.num_spks == 3:
                        targets[0].append(batch.android_s3_sig)
                        targets[1].append(batch.condenser_s3_sig)
                        targets[2].append(batch.ios_s3_sig)
                        targets[3].append(batch.lavalier_s3_sig)
                        targets[4].append(batch.XYH6X_s3_sig)
                        targets[5].append(batch.XYH6Y_s3_sig)
            
                    mix_sig = mixtures[self.channel_order[self.hparams.test_channel]]
                    mixture, mix_len = mix_sig
                    snt_id = batch.id
                    targets = targets[self.channel_order[self.hparams.test_channel]]

                    with torch.no_grad():
                        predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids = self.compute_forward(
                            [mix_sig], [targets], sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr, _, _, _ = self.compute_objectives(
                        predictions, targets, ori_targets, pred_mix, mix, chan_preds, chan_ids
                    )
                    sisnr = sisnr[0]

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets[0].device)
                    sisnr_baseline, _, _, _ = self.compute_objectives(
                        [mixture_signal], targets, ori_targets, pred_mix, mix, chan_preds, chan_ids
                    )
                    sisnr_i = sisnr - sisnr_baseline[0]

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0][0].t().cpu().numpy(),
                        predictions[0][0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0][0].t().cpu().numpy(),
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

    def save_audio(self, snt_id, mixture, targets, predictions, test_channel):
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
                save_path, "item{}_{}_source{}hat.wav".format(snt_id, test_channel, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_{}_source{}.wav".format(snt_id, test_channel, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_{}_mix.wav".format(snt_id, test_channel))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


    def tsne(self, test_data):
        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.embedloader_opts
        )
        
        embed_dict = [ [], [], [], [], [], []]
        
        with tqdm(test_loader, dynamic_ncols=True) as t:
            for i, batch in enumerate(t):

                # Apply Separation
                mixtures = [
                    batch.android_mix_sig, 
                    batch.condenser_mix_sig, 
                    batch.ios_mix_sig,
                    batch.lavalier_mix_sig,
                    batch.XYH6X_mix_sig,
                    batch.XYH6Y_mix_sig,
                ]

                for i in range(len(mixtures)):
                    mix = mixtures[i]
                    mix, mix_lens = mix
                    mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)
                    
                    with torch.no_grad():
                        chan_mix_w = self.hparams.Encoder(mix)
                        C_channel = self.hparams.ChannelEncoder(mix, chan_mix_w)
                    
                    for C in C_channel:
                        embed_dict[i].append(C.detach().cpu().numpy())
        
        ## how to tsne
        from sklearn import manifold
        n_sample = len(embed_dict[0])
        embed_dict = np.array(embed_dict)
        embed_dict = np.concatenate((embed_dict[0], embed_dict[1], embed_dict[2], embed_dict[3], embed_dict[4], embed_dict[5]), axis = 0)
        embed_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(embed_dict)
        x_min, x_max = embed_tsne.min(0), embed_tsne.max(0)
        embed_norm = (embed_tsne - x_min) / (x_max - x_min)
        colormap = plt.cm.get_cmap('tab10', 6)
        color_cycle= itertools.cycle([colormap(i) for i in range(6)])
        marker_list = ['8', 's', 'p', 'H', 'D', '^']
        label_list = ['Android', 'condenser', 'iOS', 'lavalier', 'XYH-6-X', 'XYH-6-Y']
        color = next(color_cycle)
        current_order = 0
        for i in range(len(embed_norm)):
            order = i // n_sample
            if order > current_order:
                color = next(color_cycle)
                current_order += 1
            if i % n_sample == 0:
                plt.scatter(embed_norm[i][0], embed_norm[i][1], s=20, marker=marker_list[order], color=color, label=label_list[order])
            else:
                plt.scatter(embed_norm[i][0], embed_norm[i][1], s=20, marker=marker_list[order], color=color)
        
        mpl.rc('font',family='Times New Roman')
        # mpl.rc('figure', figsize=[50, 50])

        plt.rcParams.update({'font.size': 10})
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = "Times New Roman"
        plt.legend(loc='lower right',bbox_to_anchor=(1.01, -0.01), fontsize=7)
        # plt.title(f'pairwise PCA')
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],family='Times New Roman', fontsize=8)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],family='Times New Roman', fontsize=8)
        plt.show()
        plt.savefig(os.path.join(self.hparams.save_folder, 'tsne.pdf'))
                
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

    @sb.utils.data_pipeline.takes("android_mix_wav")
    @sb.utils.data_pipeline.provides("android_mix_sig")
    def audio_pipeline_android_mix(android_mix_wav):
        android_mix_sig = sb.dataio.dataio.read_audio(android_mix_wav)
        return android_mix_sig

    @sb.utils.data_pipeline.takes("android_s1_wav")
    @sb.utils.data_pipeline.provides("android_s1_sig")
    def audio_pipeline_android_s1(android_s1_wav):
        android_s1_sig = sb.dataio.dataio.read_audio(android_s1_wav)
        return android_s1_sig

    @sb.utils.data_pipeline.takes("android_s2_wav")
    @sb.utils.data_pipeline.provides("android_s2_sig")
    def audio_pipeline_android_s2(android_s2_wav):
        android_s2_sig = sb.dataio.dataio.read_audio(android_s2_wav)
        return android_s2_sig
    
    @sb.utils.data_pipeline.takes("condenser_mix_wav")
    @sb.utils.data_pipeline.provides("condenser_mix_sig")
    def audio_pipeline_condenser_mix(condenser_mix_wav):
        condenser_mix_sig = sb.dataio.dataio.read_audio(condenser_mix_wav)
        return condenser_mix_sig

    @sb.utils.data_pipeline.takes("condenser_s1_wav")
    @sb.utils.data_pipeline.provides("condenser_s1_sig")
    def audio_pipeline_condenser_s1(condenser_s1_wav):
        condenser_s1_sig = sb.dataio.dataio.read_audio(condenser_s1_wav)
        return condenser_s1_sig

    @sb.utils.data_pipeline.takes("condenser_s2_wav")
    @sb.utils.data_pipeline.provides("condenser_s2_sig")
    def audio_pipeline_condenser_s2(condenser_s2_wav):
        condenser_s2_sig = sb.dataio.dataio.read_audio(condenser_s2_wav)
        return condenser_s2_sig
    
    @sb.utils.data_pipeline.takes("ios_mix_wav")
    @sb.utils.data_pipeline.provides("ios_mix_sig")
    def audio_pipeline_ios_mix(ios_mix_wav):
        ios_mix_sig = sb.dataio.dataio.read_audio(ios_mix_wav)
        return ios_mix_sig

    @sb.utils.data_pipeline.takes("ios_s1_wav")
    @sb.utils.data_pipeline.provides("ios_s1_sig")
    def audio_pipeline_ios_s1(ios_s1_wav):
        ios_s1_sig = sb.dataio.dataio.read_audio(ios_s1_wav)
        return ios_s1_sig

    @sb.utils.data_pipeline.takes("ios_s2_wav")
    @sb.utils.data_pipeline.provides("ios_s2_sig")
    def audio_pipeline_ios_s2(ios_s2_wav):
        ios_s2_sig = sb.dataio.dataio.read_audio(ios_s2_wav)
        return ios_s2_sig
    
    @sb.utils.data_pipeline.takes("lavalier_mix_wav")
    @sb.utils.data_pipeline.provides("lavalier_mix_sig")
    def audio_pipeline_lavalier_mix(lavalier_mix_wav):
        lavalier_mix_sig = sb.dataio.dataio.read_audio(lavalier_mix_wav)
        return lavalier_mix_sig

    @sb.utils.data_pipeline.takes("lavalier_s1_wav")
    @sb.utils.data_pipeline.provides("lavalier_s1_sig")
    def audio_pipeline_lavalier_s1(lavalier_s1_wav):
        lavalier_s1_sig = sb.dataio.dataio.read_audio(lavalier_s1_wav)
        return lavalier_s1_sig

    @sb.utils.data_pipeline.takes("lavalier_s2_wav")
    @sb.utils.data_pipeline.provides("lavalier_s2_sig")
    def audio_pipeline_lavalier_s2(lavalier_s2_wav):
        lavalier_s2_sig = sb.dataio.dataio.read_audio(lavalier_s2_wav)
        return lavalier_s2_sig
    
    @sb.utils.data_pipeline.takes("XYH6X_mix_wav")
    @sb.utils.data_pipeline.provides("XYH6X_mix_sig")
    def audio_pipeline_XYH6X_mix(XYH6X_mix_wav):
        XYH6X_mix_sig = sb.dataio.dataio.read_audio(XYH6X_mix_wav)
        return XYH6X_mix_sig

    @sb.utils.data_pipeline.takes("XYH6X_s1_wav")
    @sb.utils.data_pipeline.provides("XYH6X_s1_sig")
    def audio_pipeline_XYH6X_s1(XYH6X_s1_wav):
        XYH6X_s1_sig = sb.dataio.dataio.read_audio(XYH6X_s1_wav)
        return XYH6X_s1_sig

    @sb.utils.data_pipeline.takes("XYH6X_s2_wav")
    @sb.utils.data_pipeline.provides("XYH6X_s2_sig")
    def audio_pipeline_XYH6X_s2(XYH6X_s2_wav):
        XYH6X_s2_sig = sb.dataio.dataio.read_audio(XYH6X_s2_wav)
        return XYH6X_s2_sig
    
    @sb.utils.data_pipeline.takes("XYH6Y_mix_wav")
    @sb.utils.data_pipeline.provides("XYH6Y_mix_sig")
    def audio_pipeline_XYH6Y_mix(XYH6Y_mix_wav):
        XYH6Y_mix_sig = sb.dataio.dataio.read_audio(XYH6Y_mix_wav)
        return XYH6Y_mix_sig

    @sb.utils.data_pipeline.takes("XYH6Y_s1_wav")
    @sb.utils.data_pipeline.provides("XYH6Y_s1_sig")
    def audio_pipeline_XYH6Y_s1(XYH6Y_s1_wav):
        XYH6Y_s1_sig = sb.dataio.dataio.read_audio(XYH6Y_s1_wav)
        return XYH6Y_s1_sig

    @sb.utils.data_pipeline.takes("XYH6Y_s2_wav")
    @sb.utils.data_pipeline.provides("XYH6Y_s2_sig")
    def audio_pipeline_XYH6Y_s2(XYH6Y_s2_wav):
        XYH6Y_s2_sig = sb.dataio.dataio.read_audio(XYH6Y_s2_wav)
        return XYH6Y_s2_sig

    if hparams["num_spks"] == 3:

        @sb.utils.data_pipeline.takes("android_s3_wav")
        @sb.utils.data_pipeline.provides("android_s3_sig")
        def audio_pipeline_android_s3(android_s3_wav):
            android_s3_sig = sb.dataio.dataio.read_audio(android_s3_wav)
            return android_s3_sig
        
        @sb.utils.data_pipeline.takes("condenser_s3_wav")
        @sb.utils.data_pipeline.provides("condenser_s3_sig")
        def audio_pipeline_condenser_s3(condenser_s3_wav):
            condenser_s3_sig = sb.dataio.dataio.read_audio(condenser_s3_wav)
            return condenser_s3_sig
        
        @sb.utils.data_pipeline.takes("ios_s3_wav")
        @sb.utils.data_pipeline.provides("ios_s3_sig")
        def audio_pipeline_ios_s3(ios_s3_wav):
            ios_s3_sig = sb.dataio.dataio.read_audio(ios_s3_wav)
            return ios_s3_sig
        
        @sb.utils.data_pipeline.takes("lavalier_s3_wav")
        @sb.utils.data_pipeline.provides("lavalier_s3_sig")
        def audio_pipeline_lavalier_s3(lavalier_s3_wav):
            lavalier_s3_sig = sb.dataio.dataio.read_audio(lavalier_s3_wav)
            return lavalier_s3_sig
        
        @sb.utils.data_pipeline.takes("XYH6X_s3_wav")
        @sb.utils.data_pipeline.provides("XYH6X_s3_sig")
        def audio_pipeline_XYH6X_s3(XYH6X_s3_wav):
            XYH6X_s3_sig = sb.dataio.dataio.read_audio(XYH6X_s3_wav)
            return XYH6X_s3_sig
        
        @sb.utils.data_pipeline.takes("XYH6Y_s3_wav")
        @sb.utils.data_pipeline.provides("XYH6Y_s3_sig")
        def audio_pipeline_XYH6Y_s3(XYH6Y_s3_wav):
            XYH6Y_s3_sig = sb.dataio.dataio.read_audio(XYH6Y_s3_wav)
            return XYH6Y_s3_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_android_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_android_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_android_s2)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_condenser_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_condenser_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_condenser_s2)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_ios_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_ios_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_ios_s2)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_lavalier_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_lavalier_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_lavalier_s2)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_XYH6X_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_XYH6X_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_XYH6X_s2)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_XYH6Y_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_XYH6Y_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_XYH6Y_s2)
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_android_s3)
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_condenser_s3)
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_ios_s3)
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_lavalier_s3)
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_XYH6X_s3)
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_XYH6Y_s3)
        sb.dataio.dataset.set_output_keys(
            datasets, 
            [
                "id", 
                "android_mix_sig", 
                "android_s1_sig", 
                "android_s2_sig", 
                "android_s3_sig",
                "condenser_mix_sig", 
                "condenser_s1_sig", 
                "condenser_s2_sig", 
                "condenser_s3_sig",
                "ios_mix_sig", 
                "ios_s1_sig", 
                "ios_s2_sig", 
                "ios_s3_sig",
                "lavalier_mix_sig", 
                "lavalier_s1_sig", 
                "lavalier_s2_sig", 
                "lavalier_s3_sig",
                "XYH6X_mix_sig", 
                "XYH6X_s1_sig", 
                "XYH6X_s2_sig", 
                "XYH6X_s3_sig",
                "XYH6Y_mix_sig", 
                "XYH6Y_s1_sig", 
                "XYH6Y_s2_sig", 
                "XYH6Y_s3_sig",
            ]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, 
            [
                "id", 
                "android_mix_sig", 
                "android_s1_sig", 
                "android_s2_sig", 
                "condenser_mix_sig", 
                "condenser_s1_sig", 
                "condenser_s2_sig", 
                "ios_mix_sig", 
                "ios_s1_sig", 
                "ios_s2_sig", 
                "lavalier_mix_sig", 
                "lavalier_s1_sig", 
                "lavalier_s2_sig", 
                "XYH6X_mix_sig", 
                "XYH6X_s1_sig", 
                "XYH6X_s2_sig", 
                "XYH6Y_mix_sig", 
                "XYH6Y_s1_sig", 
                "XYH6Y_s2_sig", 
            ]
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
    
    # random seed
    random.seed(hparams['seed'])

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
    from recipes.channel_adapt.prepare_data import prepare_wsjmix  # noqa

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
            test_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
            embed_loader_kwargs=hparams["embedloader_opts"],
        )

    # Eval
    separator.evaluate(test_data, min_key="si-snr", test_loader_kwargs=hparams["dataloader_opts"])
    separator.save_results(test_data)
    separator.tsne(test_data)