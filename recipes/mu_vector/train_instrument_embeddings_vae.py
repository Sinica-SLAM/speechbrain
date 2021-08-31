#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import time
from tqdm import tqdm


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        super(SpeakerBrain, self).__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )
        self.count = 0

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
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    t.set_postfix(**self.avg_train_loss)

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

    def fit_batch(self, batch):
        if self.count % self.hparams.accum_grad_count == 0:
            self.optimizer.zero_grad()

        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.check_gradients(loss):
                self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            if isinstance(loss, dict):
                sum_loss = sum((loss.values()))
            else:
                sum_loss = loss
            self.count += 1
            sum_loss.backward()
            if (
                self.check_gradients(sum_loss)
                and self.count % self.hparams.accum_grad_count == 0
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.count = 0

        if isinstance(loss, dict):
            for key in loss.keys():
                loss[key] = loss[key].detach().cpu()
        return loss

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
                    avg_loss[key] -= avg_loss[key] / self.step
                    avg_loss[key] += float(loss[key]) / self.step
        else:
            if torch.isfinite(loss):
                avg_loss -= avg_loss / self.step
                avg_loss += float(loss) / self.step
        return avg_loss

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings, z_mean, z_logvar, bnf = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)
        reconst_feats, mu_vector = self.modules.vae_decoder(bnf)

        return (
            outputs,
            z_mean,
            z_logvar,
            bnf,
            reconst_feats,
            feats,
            lens,
            mu_vector,
        )

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """

        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        if isinstance(loss, dict):
            loss = {key: value for key, value in loss.items()}
        return loss

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        (
            predictions,
            z_mean,
            z_logvar,
            bnf,
            reconst_feats,
            feats,
            lens,
            mu_vector,
        ) = predictions
        sound_id = batch.id
        inst_id, _ = batch.inst_id_encoded
        inst_family, _ = batch.inst_family_encoded

        # Concatenate labels (due to data augmentation)
        id_loss = 0
        if stage == sb.Stage.TRAIN:
            inst_id = torch.cat([inst_id] * self.n_augment, dim=0)
            inst_family = torch.cat([inst_family] * self.n_augment, dim=0)
            id_loss = self.hparams.compute_cost(
                predictions[:, :, : self.hparams.number_instruments],
                inst_id,
                lens,
            )

        family_loss = self.hparams.compute_cost(
            predictions[:, :, self.hparams.number_instruments :],
            inst_family,
            lens,
        )

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                sound_id,
                predictions[:, :, self.hparams.number_instruments :],
                inst_family,
                lens,
            )
            self.f1_metrics.append(
                predictions[:, :, self.hparams.number_instruments :],
                inst_family,
            )

        ZERO = torch.zeros_like(z_mean)
        distq = torch.distributions.normal.Normal(
            z_mean, torch.exp(z_logvar) ** (1 / 2)
        )
        distp = torch.distributions.normal.Normal(
            ZERO, torch.exp(ZERO) ** (1 / 2)
        )
        kl = torch.distributions.kl.kl_divergence(distq, distp)
        kl = kl.mean(axis=1).mean(axis=1).mean()
        l2_loss = (
            self.hparams.reconst_alpha
            * torch.sqrt(torch.mean((reconst_feats - feats) ** 2))
            * 0.5
        )
        l1_loss = (
            self.hparams.reconst_alpha
            * torch.mean(torch.abs(reconst_feats - feats))
            * 0.5
        )

        return {
            "id": id_loss,
            "fam": family_loss,
            "kl": kl,
            "l1": l1_loss,
            "l2": l2_loss,
        }

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            self.f1_metrics = self.hparams.f1_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")
            stage_stats["F1-score"] = self.f1_metrics.summarize()

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]
    inst_label_encoder = sb.dataio.encoder.CategoricalEncoder()
    family_label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample - 1)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("inst_id", "inst_family")
    @sb.utils.data_pipeline.provides(
        "inst_id", "inst_id_encoded", "inst_family", "inst_family_encoded"
    )
    def label_pipeline(inst_id, inst_family):

        yield inst_id
        if inst_id in inst_ids:
            inst_id_encoded = inst_label_encoder.encode_sequence_torch(
                [inst_id]
            )
        else:
            inst_id_encoded = torch.Tensor([0])

        yield inst_id_encoded
        yield inst_family
        inst_family_encoded = family_label_encoder.encode_sequence_torch(
            [inst_family]
        )
        yield inst_family_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    inst_lab_enc_file = os.path.join(
        hparams["save_folder"], "inst_label_encoder.txt"
    )
    inst_label_encoder.load_or_create(
        path=inst_lab_enc_file,
        from_didatasets=[train_data],
        output_key="inst_id",
    )

    inst_ids = list(
        inst_label_encoder.from_saved(path=inst_lab_enc_file).lab2ind.keys()
    )

    family_lab_enc_file = os.path.join(
        hparams["save_folder"], "family_label_encoder.txt"
    )

    family_label_encoder.load_or_create(
        path=family_lab_enc_file,
        from_didatasets=[train_data],
        output_key="inst_family",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "inst_id_encoded", "inst_family_encoded"]
    )

    return (
        train_data,
        valid_data,
        test_data,
        inst_label_encoder,
        family_label_encoder,
    )


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from data_prepare import prepare_nsynth  # noqa

    run_on_main(
        prepare_nsynth,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "meta_train": hparams["meta_train"],
            "meta_valid": hparams["meta_valid"],
            "meta_test": hparams["meta_test"],
            "splits": ["train", "dev", "test"],
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, _, _ = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Identification test
    speaker_brain.evaluate(
        test_set=test_data, test_loader_kwargs=hparams["dataloader_options"],
    )
