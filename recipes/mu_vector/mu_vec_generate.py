import os
import sys
import random
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
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
        self.mu_vector_list = []
        self.mu_vector = []
        self.sound_id = ""
        self.last_batch = False

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
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):

                # Check if it is the last data
                if batch.id[0] == test_set.dataset.data_ids[-1]:
                    self.last_batch = True

                self.step += 1
                loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[sb.Stage.TEST, avg_test_loss, None]
            )
        self.step = 0

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Calculate Mean from stereo to mono
        wavs = torch.mean(wavs, dim=-1, keepdim=True).squeeze(-1)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens, embeddings

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using instrument-id as label.
        """
        predictions, lens, embeddings = predictions
        sound_id = batch.id
        inst_id, _ = batch.inst_id_encoded

        if not self.sound_id:
            self.sound_id = sound_id[0].split("_")[0]

        # Append the "mean" mu-vector for a track
        if self.mu_vector and self.sound_id not in sound_id[0]:
            self.mu_vector_list.append(
                {
                    self.sound_id: torch.mean(
                        torch.cat(self.mu_vector, dim=0), dim=0
                    )
                }
            )
            self.mu_vector = []
            self.sound_id = sound_id[0].split("_")[0]

        # Append mu_vectors by segment
        self.mu_vector.append(embeddings)

        # Append the last track
        if self.last_batch:
            self.mu_vector_list.append(
                {
                    self.sound_id: torch.mean(
                        torch.cat(self.mu_vector, dim=0), dim=0
                    )
                }
            )

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            inst_id = torch.cat([inst_id] * self.n_augment, dim=0)

        loss = self.hparams.compute_cost(predictions, inst_id, lens)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                sound_id, predictions, inst_id, lens,
            )
            self.f1_metrics.append(
                predictions, inst_id,
            )

        return loss

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
    xumx_train_output = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["xumx_train_annotation"],
        replacements={"data_root": data_folder},
    )

    xumx_valid_output = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["xumx_valid_annotation"],
        replacements={"data_root": data_folder},
    )

    xumx_test_output = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["xumx_test_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [xumx_train_output, xumx_valid_output, xumx_test_output]
    inst_label_encoder = sb.dataio.encoder.CategoricalEncoder()

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
    @sb.utils.data_pipeline.takes("inst_id")
    @sb.utils.data_pipeline.provides("inst_id", "inst_id_encoded")
    def label_pipeline(inst_id):
        yield inst_id
        inst_id_encoded = inst_label_encoder.encode_sequence_torch([inst_id])
        yield inst_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    inst_lab_enc_file = os.path.join(
        hparams["save_folder"], "inst_label_encoder.txt"
    )
    inst_label_encoder.load_or_create(
        path=inst_lab_enc_file,
        from_didatasets=[xumx_train_output],
        output_key="inst_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "inst_id_encoded"]
    )

    return [xumx_train_output, xumx_valid_output, xumx_test_output]


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

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from mu_vec_prepare import prepare_xmux_output  # noqa

    run_on_main(
        prepare_xmux_output,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "meta_file": hparams["meta_file"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    xumx_outputs = dataio_prep(hparams)

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

    # Generate mu-vector
    for xumx_output, split in zip(
        xumx_outputs,
        ["train_mu_vectors.pt", "valid_mu_vectors.pt", "test_mu_vectors.pt"],
    ):
        speaker_brain.evaluate(
            test_set=xumx_output,
            min_key="ErrorRate",
            test_loader_kwargs=hparams["dataloader_opts"],
        )

        # Save mu-vectors
        torch.save(
            speaker_brain.mu_vector_list,
            os.path.join(hparams["save_folder"], split),
        )
        speaker_brain.mu_vector_list = []
        speaker_brain.mu_vector = []
        speaker_brain.sound_id = ""
        speaker_brain.last_batch = False
