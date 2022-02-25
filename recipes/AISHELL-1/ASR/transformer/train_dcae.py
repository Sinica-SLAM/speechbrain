#!/usr/bin/env/python3
"""

AISHELL-1 DCAE model recipe. (Adapted from the LibriSpeech recipe.)

"""

import os
import sys
import time
import torch
import logging
import speechbrain as sb

from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main

from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    err_msg = (
        """
        The optional dependency tensorboard must be installed to run this recipe.\n
        """
        """
        Install using `pip install tensorboard`.\n
        """
    )
    raise ImportError(err_msg)

try:
    import matplotlib.pyplot as plt
except ImportError:
    err_msg = (
        """
        The optional dependency matplotlib must be installed to run this recipe.\n
        """
        """
        Install using `pip install matplotlib`.\n
        """
    )
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        super().__init__(
            modules, opt_class, hparams, run_opts, checkpointer,
        )

        self.avg_train_loss_l1 = 0
        self.avg_train_loss_l2 = 0
        self.avg_train_loss_ctc = 0
        self.avg_train_loss_seq = 0

        # Tensorboard initialization
        tensorboard_path = self.hparams.output_folder + "/tensorboard"
        self.summary_writer = SummaryWriter(log_dir=tensorboard_path)
        self.fbanks = []
        self.train_fbank_ids = []
        self.valid_fbank_ids = []

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        batch_size = wavs.size()[0]

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # Prepare ground-truth FBank for reconstruction loss
        original_feats = feats
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                original_feats = feats[:batch_size, :, :]
                original_feats = original_feats.repeat(2, 1, 1)

        # forward modules
        enc_out, reconstructed_feats, pred = self.hparams.DCAE(
            feats, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # output layer for reconstruction loss
        reconstructed_feats = self.modules.rec_lin(reconstructed_feats)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps, reconstructed_feats, original_feats

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (
            p_ctc,
            p_seq,
            wav_lens,
            hyps,
            reconstructed_feats,
            feats,
        ) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss_l1 = self.hparams.l1_cost(reconstructed_feats, feats, wav_lens)
        loss_l2 = self.hparams.l2_cost(reconstructed_feats, feats, wav_lens)

        reconstruction_loss = self.hparams.l1_weight * loss_l1 + loss_l2
        asr_loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )
        loss = (
            self.hparams.reconstruction_weight * reconstruction_loss + asr_loss
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                if self.hparams.remove_spaces:
                    predicted_words = ["".join(p) for p in predicted_words]
                    target_words = ["".join(t) for t in target_words]
                    self.cer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        # Add FBank images to tensorboard
        # TODO(jamfly): kinda messy, need to come up with a smarter/clean way
        if stage == sb.Stage.VALID or stage == sb.Stage.TRAIN:
            fbank_ids = (
                self.valid_fbank_ids
                if stage == sb.Stage.VALID
                else self.train_fbank_ids
            )
            feats = feats.cpu().detach()
            reconstructed_feats = reconstructed_feats.cpu().detach()
            fbanks = zip(ids, feats, reconstructed_feats)

            # Random pick num_figs samples from batch
            if len(fbank_ids) == 0:
                num_left_fbanks = self.hparams.num_figs // 2
                num_right_fbanks = self.hparams.num_figs - num_left_fbanks
                left_fbank_ids = ids[:num_left_fbanks]
                right_fbank_ids = ids[-num_right_fbanks:]
                fbank_ids += left_fbank_ids
                fbank_ids += right_fbank_ids

            for fbank in fbanks:
                # When already found our target ids, skip finding
                if len(self.fbanks) == self.hparams.num_figs:
                    break

                if fbank[0] in fbank_ids:
                    self.fbanks.append(fbank)

        return loss_l1, loss_l2, loss_ctc, loss_seq, loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss_l1, loss_l2, loss_ctc, loss_seq, loss = self.compute_objectives(
            predictions, batch, sb.Stage.TRAIN
        )

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return (
            loss_l1.detach(),
            loss_l2.detach(),
            loss_ctc.detach(),
            loss_seq.detach(),
            loss.detach(),
        )

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
                global_steps = (epoch - 1) * len(t)
                for batch in t:
                    if self._optimizer_step_limit_exceeded:
                        logger.info("Train iteration limit exceeded")
                        break
                    self.step += 1
                    global_steps += 1

                    loss_l1, loss_l2, loss_ctc, loss_seq, loss = self.fit_batch(
                        batch
                    )
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    self.avg_train_loss_seq = self.update_average(
                        loss_seq, self.avg_train_loss_seq
                    )
                    self.avg_train_loss_ctc = self.update_average(
                        loss_ctc, self.avg_train_loss_ctc
                    )
                    self.avg_train_loss_l2 = self.update_average(
                        loss_l2, self.avg_train_loss_l2
                    )
                    self.avg_train_loss_l1 = self.update_average(
                        loss_l1, self.avg_train_loss_l1
                    )

                    t.set_postfix(
                        l1=self.avg_train_loss_l1,
                        l2=self.avg_train_loss_l2,
                        ctc_loss=self.avg_train_loss_ctc,
                        seq_loss=self.avg_train_loss_seq,
                        train_loss=self.avg_train_loss,
                    )

                    # Add losses to tensorboard
                    self.summary_writer.add_scalar(
                        "Train/loss", loss, global_steps
                    )
                    self.summary_writer.add_scalar(
                        "Train/l1", loss_l1, global_steps
                    )
                    self.summary_writer.add_scalar(
                        "Train/l2", loss_l2, global_steps
                    )
                    self.summary_writer.add_scalar(
                        "Train/ctc_loss", loss_ctc, global_steps,
                    )
                    self.summary_writer.add_scalar(
                        "Train/attention_loss", loss_seq, global_steps,
                    )

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        # This should not use run_on_main, because that
                        # includes a DDP barrier. That eventually leads to a
                        # crash when the processes'
                        # time.time() - last_ckpt_time differ and some
                        # processes enter this block while others don't,
                        # missing the barrier.
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(
                sb.Stage.TRAIN,
                self.avg_train_loss,
                self.avg_train_loss_seq,
                self.avg_train_loss_ctc,
                self.avg_train_loss_l2,
                self.avg_train_loss_l1,
                epoch,
            )
            self.avg_train_loss = 0.0
            self.avg_train_loss_seq = 0.0
            self.avg_train_loss_ctc = 0.0
            self.avg_train_loss_l2 = 0.0
            self.avg_train_loss_l1 = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()

                avg_valid_loss = 0.0
                avg_valid_loss_seq = 0.0
                avg_valid_loss_ctc = 0.0
                avg_valid_loss_l2 = 0.0
                avg_valid_loss_l1 = 0.0

                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        (
                            loss_l1,
                            loss_l2,
                            loss_ctc,
                            loss_seq,
                            loss,
                        ) = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )
                        avg_valid_loss_seq = self.update_average(
                            loss_seq, avg_valid_loss_seq
                        )
                        avg_valid_loss_ctc = self.update_average(
                            loss_ctc, avg_valid_loss_ctc
                        )
                        avg_valid_loss_l2 = self.update_average(
                            loss_l2, avg_valid_loss_l2
                        )
                        avg_valid_loss_l1 = self.update_average(
                            loss_l1, avg_valid_loss_l1
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[
                            sb.Stage.VALID,
                            avg_valid_loss,
                            avg_valid_loss_seq,
                            avg_valid_loss_ctc,
                            avg_valid_loss_l2,
                            avg_valid_loss_l1,
                            epoch,
                        ],
                    )

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            (
                loss_l1,
                loss_l2,
                loss_ctc,
                loss_seq,
                loss,
            ) = self.compute_objectives(predictions, batch, stage)

        return (
            loss_l1.detach(),
            loss_l2.detach(),
            loss_ctc.detach(),
            loss_seq.detach(),
            loss.detach(),
        )

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        avg_test_loss = 0.0
        avg_test_loss_seq = 0.0
        avg_test_loss_ctc = 0.0
        avg_test_loss_l2 = 0.0
        avg_test_loss_l1 = 0.0

        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                (
                    loss_l1,
                    loss_l2,
                    loss_ctc,
                    loss_seq,
                    loss,
                ) = self.evaluate_batch(batch, stage=sb.Stage.TEST)

                avg_test_loss = self.update_average(loss, avg_test_loss)
                avg_test_loss_seq = self.update_average(
                    loss_seq, avg_test_loss_seq
                )
                avg_test_loss_ctc = self.update_average(
                    loss_ctc, avg_test_loss_ctc
                )
                avg_test_loss_l2 = self.update_average(
                    loss_l2, avg_test_loss_l2
                )
                avg_test_loss_l1 = self.update_average(
                    loss_l1, avg_test_loss_l1
                )

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end,
                args=[
                    sb.Stage.TEST,
                    avg_test_loss,
                    avg_test_loss_seq,
                    avg_test_loss_ctc,
                    avg_test_loss_l2,
                    avg_test_loss_l1,
                    None,
                ],
            )
        self.step = 0
        return avg_test_loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage == sb.Stage.TRAIN or stage == sb.Stage.VALID:
            self.fbanks = []

        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(
        self, stage, stage_loss, loss_seq, loss_ctc, loss_l2, loss_l1, epoch,
    ):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {
            "l1": loss_l1,
            "l2": loss_l2,
            "ctc_loss": loss_ctc,
            "seq_loss": loss_seq,
            "loss": stage_loss,
        }

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

            for uid, fbank, re_fbank in self.fbanks:
                fbank_fig_path = (
                    f"""{self.hparams.output_folder}/tensorboard"""
                    f"""/fbank/train/{uid}"""
                )
                re_fbank_fig_path = (
                    f"{self.hparams.output_folder}/tensorboard"
                    ""
                    f"""/re_fbank/train/{uid}"""
                )
                self._plot_mel_fbank(
                    uid, fbank, fbank_fig_path, f"Train/FBank/{uid}", epoch,
                )
                self._plot_mel_fbank(
                    uid,
                    re_fbank,
                    re_fbank_fig_path,
                    f"Train/ReFBank/{uid}",
                    epoch,
                )
            self.summary_writer.close()
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            # Add losses / FBanks to tensorboard
            if sb.Stage.VALID:
                self.summary_writer.add_scalar("Valid/loss", stage_loss, epoch)
                self.summary_writer.add_scalar("Valid/l1", loss_l1, epoch)
                self.summary_writer.add_scalar("Valid/l2", loss_l2, epoch)
                self.summary_writer.add_scalar(
                    "Valid/ctc_loss", loss_ctc, epoch
                )
                self.summary_writer.add_scalar(
                    "Valid/attention_loss", loss_seq, epoch,
                )
                self.summary_writer.add_scalar(
                    "Valid/accuracy", stage_stats["ACC"], epoch
                )
                for uid, fbank, re_fbank in self.fbanks:
                    fbank_fig_path = (
                        f"""{self.hparams.output_folder}/tensorboard"""
                        f"""/fbank/valid/{uid}"""
                    )
                    re_fbank_fig_path = (
                        f"{self.hparams.output_folder}/tensorboard"
                        ""
                        f"""/re_fbank/valid/{uid}"""
                    )
                    self._plot_mel_fbank(
                        uid, fbank, fbank_fig_path, f"Valid/FBank/{uid}", epoch,
                    )
                    self._plot_mel_fbank(
                        uid,
                        re_fbank,
                        re_fbank_fig_path,
                        f"Valid/ReFBank/{uid}",
                        epoch,
                    )
                self.summary_writer.close()
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

                if stage == sb.Stage.VALID:
                    self.summary_writer.add_scalar(
                        "Valid/CER", stage_stats["CER"], epoch
                    )
                    self.summary_writer.close()

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=10,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Initialize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitialize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load latest checkpoint to resume training if interrupted
            if self.checkpointer is not None:

                # do not reload the weights if training is interrupted right before stage 2
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return

                self.checkpointer.recover_if_possible(
                    device=torch.device(self.device)
                )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def _plot_mel_fbank(self, uid, fbank, path, tag, epoch):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(uid)
        axs.imshow(fbank, aspect="auto")
        axs.set_ylabel("frequency bin")
        axs.set_xlabel("mel bin")

        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        fbank_fig_name = f"{path}/{epoch}.png"
        fig.savefig(fbank_fig_name)
        self.summary_writer.add_figure(
            tag, fig, epoch,
        )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass
        #  train_data = train_data.filtered_sorted(
        #  sort_key="duration",
        #  key_min_value={"duration": 2},
        #  key_max_value={"duration": 3},
        #  )

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # Defining tokenizer and loading it
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # 1.  # Dataset prep (parsing Librispeech)
    from prepare import prepare_aishell  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_aishell,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data, tokenizer = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    #  asr_brain.evaluate(
    #  test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    #  )
