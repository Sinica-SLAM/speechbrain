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
 * Y.W. Chen
"""

import sys
import torch
import torch.nn as nn
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from speechbrain.lobes.utils import center_trim, apply_model
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
import logging


# from augment import FlipChannels, FlipSign, Scale, Shift, Remix, SpeedPerturb, RandomPitch
# import json
# from torchinfo import summary

# Define training procedure
class Separation(sb.Brain):
    def __init__(  # noqa: C901
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):

        super(Separation, self).__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )

        self.count = 0
        self.model = self.hparams.model.to(self.device)
        # summary(self.model, device=self.device,
        #     col_names=("num_params", "kernel_size"))

        # Demucs augment
        # self.augment = [Shift(self.hparams.data_stride * self.hparams.sample_rate)]
        # self.augment = [RandomPitch(), SpeedPerturb()]
        # self.augment += [Shift(self.hparams.data_stride * self.hparams.sample_rate), FlipSign(), FlipChannels(), Scale(), Remix(group_size=self.hparams.remix_group_size)]
        # self.augment = nn.Sequential(*self.augment).to(self.device)
        # print(self.augment)
        self.augment = nn.Sequential().to(self.device)

    def compute_forward(self, mix, targets, stage, noise=None):

        sources = {"vocals": 0, "bass": 1, "drums": 2, "other": 3}

        """Forward computations from the mixture to the separated signals."""

        if stage == "inference":
            mix = (
                torch.from_numpy(mix).unsqueeze(0).float().to(self.device)
            )  # (B, T, C)
            ref = mix.mean(dim=-1)  # mono mixture

            # Standardize
            mix = (mix - ref.mean()) / ref.std()
            mix = mix.transpose(1, 2)  # (B, C, T)
            est_source = apply_model(
                self.model.to(self.device), mix, split=self.hparams.split
            )
            est_source = est_source * ref.std() + ref.mean()

            return est_source

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix  # (B, T, C)
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [
                targets[i][0].unsqueeze(-1)
                for i in range(self.hparams.num_sources)
            ],
            dim=-1,
        ).to(
            self.device
        )  # (B, T, C, N)

        mix = mix.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        targets = targets.permute(0, 3, 2, 1)  # (B, T, C, N) -> (B, N, C, T)

        # Add demucs augment
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                targets = self.augment(targets)
                mix = targets.sum(dim=1)  # (B, C, T)

                # for i, m in enumerate(mix):
                #     torchaudio.save("aug_audio/mix.wav", m.cpu(), 44100)

                #     for s, t in zip(self.hparams.source_names, targets[i]):
                #         torchaudio.save("aug_audio/" + s + ".wav", t.squeeze().cpu(), 44100)

        est_source = self.model(mix)  # (B, N, C, T)
        target_source = targets

        # For single source
        if len(self.hparams.source_names) == 1:
            target_source = target_source[
                :, sources[self.hparams.source_names[0]], :, :
            ].unsqueeze(1)

        target_source = center_trim(target_source, est_source)  # (B, N, C, T)

        return (
            est_source,
            target_source,
        )  # (ordering -- vocal, bass, drum, other)

    def compute_objectives(self, est_source, target_source):
        """Computes the l1 loss"""

        loss = self.hparams.loss(est_source, target_source)
        # criterion = nn.L1Loss()
        # loss = criterion(est_source, target_source)

        return loss

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [
            batch.vocals_sig,
            batch.bass_sig,
            batch.drums_sig,
            batch.other_sig,
        ]

        est_source, target_source = self.compute_forward(
            mixture, targets, sb.Stage.TRAIN
        )

        loss = self.compute_objectives(est_source, target_source)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.grad_accum_count).backward()

        # Gradient accumulation
        if self.step % self.hparams.grad_accum_count == 0:
            self.check_gradients(loss)
            grad_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm() ** 2

            grad_norm = grad_norm ** 0.5
            # print(grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        mixture = batch.mix_sig

        targets = [
            batch.vocals_sig,
            batch.bass_sig,
            batch.drums_sig,
            batch.other_sig,
        ]

        with torch.no_grad():
            est_source, target_source = self.compute_forward(
                mixture, targets, stage
            )
            loss = self.compute_objectives(est_source, target_source)

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

    def save_results(
        self, results_path, workers=2, rank=0, save=True, world_size=1
    ):

        """This script computes the SDR metrics and saves
        them into a csv file"""

        import musdb
        import museval

        # import soundfile as sf
        from scipy.io import wavfile
        import gzip
        import sys
        from concurrent import futures
        from torch import distributed
        from pathlib import Path

        self.device = "cpu"
        # we load tracks from the original musdb set
        musdb_path = "/mnt/md1/datasets/musdb18"
        test_set = musdb.DB(musdb_path, subsets=["test"], is_wav=True)

        output_dir = Path(results_path) / "results"
        output_dir.mkdir(exist_ok=True, parents=True)
        json_folder = Path(results_path) / "results/test"
        json_folder.mkdir(exist_ok=True, parents=True)

        # we load tracks from the original musdb set
        pendings = []

        with futures.ProcessPoolExecutor(workers or 1) as pool:
            for index in tqdm(
                range(rank, len(test_set), world_size), file=sys.stdout
            ):
                track = test_set.tracks[index]

                out = json_folder / f"{track.name}.json.gz"
                if out.exists():
                    continue

                # Avoid memory accumulation
                with torch.no_grad():
                    estimates = self.compute_forward(
                        track.audio, targets=None, stage="inference"
                    )

                estimates = estimates.transpose(1, 2)
                references = torch.stack(
                    [
                        torch.from_numpy(track.targets[name].audio).t()
                        for name in self.hparams.model.sources
                    ]
                )

                references = references.transpose(1, 2).cpu().numpy()
                estimates = estimates.cpu().numpy()
                win = int(1.0 * self.hparams.sample_rate)
                hop = int(1.0 * self.hparams.sample_rate)

                if save:
                    folder = Path(results_path) / "wav/test" / track.name
                    folder.mkdir(exist_ok=True, parents=True)
                    for name, estimate in zip(
                        self.hparams.model.sources, estimates
                    ):
                        wavfile.write(
                            str(folder / (name + ".wav")), 44100, estimate
                        )

                if workers:
                    pendings.append(
                        (
                            track.name,
                            pool.submit(
                                museval.evaluate,
                                references,
                                estimates,
                                win=win,
                                hop=hop,
                            ),
                        )
                    )
                else:
                    pendings.append(
                        (
                            track.name,
                            museval.evaluate(
                                references, estimates, win=win, hop=hop
                            ),
                        )
                    )
                del references, estimates, track

            for track_name, pending in tqdm(pendings, file=sys.stdout):
                if workers:
                    pending = pending.result()
                sdr, isr, sir, sar = pending
                track_store = museval.TrackStore(
                    win=44100, hop=44100, track_name=track_name
                )
                for idx, target in enumerate(self.hparams.model.sources):
                    values = {
                        "SDR": sdr[idx].tolist(),
                        "SIR": sir[idx].tolist(),
                        "ISR": isr[idx].tolist(),
                        "SAR": sar[idx].tolist(),
                    }

                    track_store.add_target(target_name=target, values=values)
                    json_path = json_folder / f"{track_name}.json.gz"
                    gzip.open(json_path, "w").write(
                        track_store.json.encode("utf-8")
                    )
            if world_size > 1:
                distributed.barrier()


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
    @sb.utils.data_pipeline.takes("mix_wav", "start", "stop", "mean", "std")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav, start, stop, mean, std):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        mix_sig, fs = torchaudio.load(
            mix_wav, num_frames=num_frames, frame_offset=start
        )
        mix_sig = mix_sig.transpose(0, 1)
        mix_sig = (mix_sig - float(mean)) / float(std)

        return mix_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)

    @sb.utils.data_pipeline.takes("vocals_wav", "start", "stop", "mean", "std")
    @sb.utils.data_pipeline.provides("vocals_sig")
    def audio_pipeline_vocals(vocals_wav, start, stop, mean, std):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        vocals_sig, fs = torchaudio.load(
            vocals_wav, num_frames=num_frames, frame_offset=start
        )
        vocals_sig = vocals_sig.transpose(0, 1)
        vocals_sig = (vocals_sig - float(mean)) / float(std)

        return vocals_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_vocals)

    # Bass data
    @sb.utils.data_pipeline.takes("bass_wav", "start", "stop", "mean", "std")
    @sb.utils.data_pipeline.provides("bass_sig")
    def audio_pipeline_bass(bass_wav, start, stop, mean, std):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        bass_sig, fs = torchaudio.load(
            bass_wav, num_frames=num_frames, frame_offset=start
        )
        bass_sig = bass_sig.transpose(0, 1)
        bass_sig = (bass_sig - float(mean)) / float(std)

        return bass_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_bass)

    # Drums data
    @sb.utils.data_pipeline.takes("drums_wav", "start", "stop", "mean", "std")
    @sb.utils.data_pipeline.provides("drums_sig")
    def audio_pipeline_drums(drums_wav, start, stop, mean, std):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        drums_sig, fs = torchaudio.load(
            drums_wav, num_frames=num_frames, frame_offset=start
        )
        drums_sig = drums_sig.transpose(0, 1)
        drums_sig = (drums_sig - float(mean)) / float(std)

        return drums_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_drums)

    # Other
    @sb.utils.data_pipeline.takes("other_wav", "start", "stop", "mean", "std")
    @sb.utils.data_pipeline.provides("other_sig")
    def audio_pipeline_other(other_wav, start, stop, mean, std):
        start = int(start)
        stop = int(stop)

        num_frames = stop - start
        other_sig, fs = torchaudio.load(
            other_wav, num_frames=num_frames, frame_offset=start
        )
        other_sig = other_sig.transpose(0, 1)
        other_sig = (other_sig - float(mean)) / float(std)

        return other_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_other)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "mean",
            "std",
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

    # Data preparation
    from recipes.MUSDB18.prepare_data import prepare_musdb18  # noqa

    # Samples adjustment
    samples = hparams["model"].valid_length(hparams["samples"])
    print(f"Number of training samples adjusted to {samples}")
    samples = samples + hparams["data_stride"]

    run_on_main(
        prepare_musdb18,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "origin_datapath": hparams["origin_datapath"],
            "target_datapath": hparams["target_datapath"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
            "data_stride": hparams["data_stride"],
            "samples": samples,
        },
    )

    # Create dataset objects
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
    separator.save_results(results_path=hparams["save_results"], save=False)
