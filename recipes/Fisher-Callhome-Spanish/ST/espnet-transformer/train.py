#!/usr/bin/env/python3
"""Recipe for training a Transformer based ST system with Fisher-Callhome.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beam search coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/conformer.yaml

Authors
 * YAO-FEI, CHENG 2021
"""

import sys
import torch
import logging

from typing import List

import speechbrain as sb

from sacremoses import MosesDetokenizer
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)
en_detoeknizer = MosesDetokenizer(lang="en")


class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        wavs, wav_lens = batch.sig

        tokens, _ = batch.tokens  # for translation task
        transcription_tokens, _ = batch.transcription_tokens  # for asr task

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # The input sizes corrsponding to ilens in ESPnet
        feature_sizes = torch.round(wav_lens * feats.shape[1]).int()

        # forward modules
        st_pred_pad = None
        asr_pred_pad = None
        loss_ctc = 0

        if stage == sb.Stage.TEST:
            from argparse import Namespace

            args = {
                "beam_size": 10,
                "penalty": 0.3,
                "maxlenratio": 0.3,
                "minlenratio": 0.0,
                "nbest": 1,
            }
            args = Namespace(**args)
            vocabs = read_vocab()

            hyps = []
            for feat in feats:
                with torch.no_grad():
                    top_b_hyps = self.hparams.Transformer.translate(
                        feat, args, vocabs
                    )
                    hyp = top_b_hyps[0]["yseq"]

                hyps.append(hyp)
        else:
            (
                enc_out,
                enc_mask,
                st_pred_pad,
                pred_mask,
            ) = self.hparams.Transformer(feats, feature_sizes, tokens)

            if self.hparams.asr_weight > 0:
                asr_pred_pad, loss_ctc = self.hparams.Transformer.forward_asr(
                    enc_out, enc_mask, transcription_tokens
                )

            # compute outputs
            if stage == sb.Stage.TRAIN:
                hyps = None
            elif stage == sb.Stage.VALID:
                hyps = enc_out.argmax(dim=-1)

        return st_pred_pad, asr_pred_pad, hyps, loss_ctc

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (st_pred_pad, asr_pred_pad, hyps, loss_ctc,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        transcription_eos, _ = batch.transcription_eos
        current_epoch = self.hparams.epoch_counter.current
        valid_search_interval = self.hparams.valid_search_interval

        loss = torch.tensor(0)
        if stage == sb.Stage.TEST:
            # 4 references bleu score
            predictions = [
                en_detoeknizer.detokenize(
                    self.hparams.tokenizer.decode_ids(utt_seq).split(" ")
                )
                for utt_seq in hyps
            ]

            four_references = [
                batch.translation_0,
                batch.translation_1,
                batch.translation_2,
                batch.translation_3,
            ]

            targets = []
            for reference in four_references:
                detokenized_translation = [
                    en_detoeknizer.detokenize(translation.split(" "))
                    for translation in reference
                ]
                targets.append(detokenized_translation)

            self.bleu_metric.append(ids, predictions, targets)
        elif stage == sb.Stage.VALID:
            if current_epoch % valid_search_interval == 0:
                predictions = [
                    en_detoeknizer.detokenize(
                        self.hparams.tokenizer.decode_ids(utt_seq).split(" ")
                    )
                    for utt_seq in hyps
                ]

                targets = [
                    en_detoeknizer.detokenize(translation.split(" "))
                    for translation in batch.translation_0
                ]
                self.bleu_metric.append(ids, predictions, [targets])

            asr_attention_loss = 0
            if self.hparams.asr_weight > 0 and self.hparams.ctc_weight < 1:
                asr_attention_loss = self.hparams.seq_cost(
                    asr_pred_pad, transcription_eos
                )
            asr_loss = (
                self.hparams.ctc_weight * loss_ctc
                + (1 - self.hparams.ctc_weight) * asr_attention_loss
            )
            st_loss = self.hparams.seq_cost(st_pred_pad, tokens_eos)
            loss = (
                self.hparams.asr_weight * asr_loss
                + (1 - self.hparams.asr_weight) * st_loss
            )

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(st_pred_pad, tokens_eos, tokens_eos_lens)
        else:
            asr_attention_loss = 0
            if self.hparams.asr_weight > 0 and self.hparams.ctc_weight < 1:
                asr_attention_loss = self.hparams.seq_cost(
                    asr_pred_pad, transcription_eos
                )
            asr_loss = (
                self.hparams.ctc_weight * loss_ctc
                + (1 - self.hparams.ctc_weight) * asr_attention_loss
            )
            st_loss = self.hparams.seq_cost(st_pred_pad, tokens_eos)
            loss = (
                self.hparams.asr_weight * asr_loss
                + (1 - self.hparams.asr_weight) * st_loss
            )

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            if stage == sb.Stage.VALID:
                stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if stage == sb.Stage.TEST:
                stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")
            elif (
                current_epoch % valid_search_interval == 0
                and stage == sb.Stage.VALID
            ):
                stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            current_epoch = self.hparams.epoch_counter.current

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
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            with open(self.hparams.bleu_file, "a+", encoding="utf-8") as w:
                self.bleu_metric.write_stats(w)

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


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def sp_audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.unsqueeze(0)
        sig = hparams["speed_perturb"](sig)
        sig = sig.squeeze(0)
        return sig

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    # The tokens without BOS or EOS is for computing CTC loss.
    @sb.utils.data_pipeline.takes("translation_0")
    @sb.utils.data_pipeline.provides(
        "translation_0", "tokens_list", "tokens_bos", "tokens_eos", "tokens",
    )
    def one_reference_text_pipeline(translation):
        """Processes the transcriptions to generate proper labels"""
        yield translation
        tokens_list = hparams["tokenizer"].encode_as_ids(translation)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    @sb.utils.data_pipeline.takes(
        "translation_0", "translation_1", "translation_2", "translation_3",
    )
    @sb.utils.data_pipeline.provides(
        "translation_0",
        "translation_1",
        "translation_2",
        "translation_3",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
    )
    def four_reference_text_pipeline(*translations):
        """Processes the transcriptions to generate proper labels"""
        yield translations[0]
        yield translations[1]
        yield translations[2]
        yield translations[3]
        tokens_list = hparams["tokenizer"].encode_as_ids(translations[0])
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    @sb.utils.data_pipeline.takes("transcription")
    @sb.utils.data_pipeline.provides(
        "transcription",
        "transcription_list",
        "transcription_bos",
        "transcription_eos",
        "transcription_tokens",
    )
    def transcription_text_pipeline(transcription):
        yield transcription
        tokens_list = hparams["tokenizer"].encode_as_ids(transcription)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    datasets = {}
    data_folder = hparams["data_folder"]
    for dataset in ["train-sp", "dev"]:
        json_path = f"{data_folder}/{dataset}/data.json"
        dataset = dataset if dataset == "train-sp" else "valid"

        is_use_sp = dataset == "train" and "speed_perturb" in hparams
        audio_pipeline_func = sp_audio_pipeline if is_use_sp else audio_pipeline

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[
                audio_pipeline_func,
                one_reference_text_pipeline,
                transcription_text_pipeline,
            ],
            output_keys=[
                "id",
                "sig",
                "duration",
                "translation_0",
                "tokens_bos",
                "tokens_eos",
                "tokens",
                "transcription",
                "transcription_list",
                "transcription_bos",
                "transcription_eos",
                "transcription_tokens",
            ],
        )

    for dataset in ["dev", "dev2", "test"]:
        json_path = f"{data_folder}/{dataset}/data.json"
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[
                audio_pipeline,
                four_reference_text_pipeline,
                transcription_text_pipeline,
            ],
            output_keys=[
                "id",
                "sig",
                "duration",
                "translation_0",
                "translation_1",
                "translation_2",
                "translation_3",
                "tokens_bos",
                "tokens_eos",
                "tokens",
                "transcription",
                "transcription_list",
                "transcription_bos",
                "transcription_eos",
                "transcription_tokens",
            ],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration"
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration"
            )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration", reverse=True
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration", reverse=True
            )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 5},
                key_max_value={"duration": 5},
                sort_key="duration",
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 5}, key_max_value={"duration": 5},
            )

        hparams["train_dataloader_opts"]["shuffle"] = True
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    return datasets


def read_vocab() -> List[str]:
    with open("../../Tokenizer/save/1000_bpe.vocab") as vocab_file:
        lines = vocab_file.readlines()
        lines = list(map(lambda line: line[0], lines))

    return lines


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # transcription/translation tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    st_brain = ST(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    st_brain.fit(
        st_brain.hparams.epoch_counter,
        datasets["train-sp"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # for dataset in ["dev", "dev2", "test"]:
    #     st_brain.evaluate(
    #         datasets[dataset],
    #         test_loader_kwargs=hparams["test_dataloader_opts"],
    #     )
