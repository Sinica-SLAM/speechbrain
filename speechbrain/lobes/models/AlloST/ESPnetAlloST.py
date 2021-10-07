#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Academia Sinica (Freddy CHENG)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Dual Encoder of Conformer speech translation model (pytorch).
It is a fusion of `e2e_st_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100
"""

from argparse import Namespace
import logging

import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask

from speechbrain.lobes.models.AlloST.ESPnetDualEncoder import DualEncoder
from speechbrain.lobes.models.transformer.ESPnetTransformer import (
    E2E as E2ETransformer,
)


class E2E(E2ETransformer):
    """E2E module.
    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        adim: int,
        phone_dim: int,
        aheads: int,
        wshare: int,
        ldconv_encoder_kernel_length: int,
        ldconv_usebias: bool,
        eunits: int,
        elayers: int,
        alayers: int,
        transformer_input_layer: str,
        transformer_encoder_selfattn_layer_type: str,
        transformer_decoder_selfattn_layer_type: str,
        ldconv_decoder_kernel_length: int,
        dunits: int,
        dlayers: int,
        transformer_encoder_pos_enc_layer_type: str,
        transformer_encoder_activation_type: str,
        macaron_style: bool = True,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        dropout_rate: float = 0.1,
        transformer_attn_dropout_rate: float = 0,
        encoder_fusion: bool = True,
        encoder_share_weights: bool = False,
        decoder_fusion_type: str = "vanilla",
        phone_embed_type: str = "embed",
        ngram: int = 3,
        global_score_weight: float = 0.1,
        down_sample_factors: int = 1,
        is_score_consensus_attention: bool = True,
        sos: int = 1,
        eos: int = 2,
        ignore_id: int = -1,
    ):
        """Construct an E2E object.
        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(
            idim,
            odim,
            adim,
            aheads,
            wshare,
            ldconv_encoder_kernel_length,
            ldconv_usebias,
            eunits,
            elayers,
            transformer_input_layer,
            transformer_encoder_selfattn_layer_type,
            transformer_decoder_selfattn_layer_type,
            ldconv_decoder_kernel_length,
            dunits,
            dlayers,
            dropout_rate,
            transformer_attn_dropout_rate,
            sos,
            eos,
            ignore_id,
        )

        self.encoder = DualEncoder(
            idim=idim,
            auxiliary_idim=phone_dim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            auxiliary_num_blocks=alayers,
            input_layer=transformer_input_layer,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            attention_dropout_rate=transformer_attn_dropout_rate,
            pos_enc_layer_type=transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=transformer_encoder_selfattn_layer_type,
            activation_type=transformer_encoder_activation_type,
            macaron_style=macaron_style,
            use_cnn_module=use_cnn_module,
            cnn_module_kernel=cnn_module_kernel,
            is_fusion=encoder_fusion,
            is_share_weights=encoder_share_weights,
            phone_embed_type=phone_embed_type,
            ngram=ngram,
            global_score_weight=global_score_weight,
            down_sample_factors=down_sample_factors,
            is_score_consensus_attention=is_score_consensus_attention,
        )

        self.is_decoder_use_extra_attn = True
        self.is_share_weights = encoder_share_weights

        if decoder_fusion_type == "stacked":
            self.decoder = Decoder(
                odim=odim,
                fusion_type="stacked",
                selfattention_layer_type=transformer_decoder_selfattn_layer_type,
                attention_dim=adim,
                attention_heads=aheads,
                conv_wshare=wshare,
                conv_kernel_length=ldconv_decoder_kernel_length,
                conv_usebias=ldconv_usebias,
                linear_units=dunits,
                num_blocks=dlayers,
                dropout_rate=dropout_rate,
                positional_dropout_rate=dropout_rate,
                self_attention_dropout_rate=transformer_attn_dropout_rate,
                src_attention_dropout_rate=transformer_attn_dropout_rate,
            )
        elif decoder_fusion_type == "gate":
            self.decoder = Decoder(
                odim=odim,
                fusion_type="gate",
                selfattention_layer_type=transformer_decoder_selfattn_layer_type,
                attention_dim=adim,
                attention_heads=aheads,
                conv_wshare=wshare,
                conv_kernel_length=ldconv_decoder_kernel_length,
                conv_usebias=ldconv_usebias,
                linear_units=dunits,
                num_blocks=dlayers,
                dropout_rate=dropout_rate,
                positional_dropout_rate=dropout_rate,
                self_attention_dropout_rate=transformer_attn_dropout_rate,
                src_attention_dropout_rate=transformer_attn_dropout_rate,
            )
        elif decoder_fusion_type == "vanilla":
            self.is_decoder_use_extra_attn = False
        else:
            AssertionError(f"not support {decoder_fusion_type} yet")

    def forward(self, xs_pad, ilens, ys_pad, ys_pad_phn, phone_lens):
        """E2E forward.
        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        ys_pad_phn = ys_pad_phn[:, : max(phone_lens)]

        src_mask = (
            make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        )
        phone_mask = (
            make_non_pad_mask(phone_lens.tolist())
            .to(ys_pad_phn.device)
            .unsqueeze(-2)
        )

        hs_pad, hs_mask, phone, phone_mask = self.encoder(
            xs_pad, src_mask, ys_pad_phn, phone_mask
        )

        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos, self.eos, self.ignore_id
        )
        ys_mask = target_mask(ys_in_pad, self.ignore_id)

        if self.is_decoder_use_extra_attn:
            pred_pad, pred_mask = self.decoder(
                ys_in_pad, ys_mask, hs_pad, hs_mask, phone, phone_mask
            )
        else:
            pred_pad, pred_mask = self.decoder(
                ys_in_pad, ys_mask, hs_pad, hs_mask
            )

        return hs_pad, hs_mask, pred_pad, pred_mask

    def translate(  # noqa: C901
        self, x, phone_sequence, trans_args, char_list=None,
    ):
        """Translate input speech.
        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param phone_sequence: phone sequence (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :return: N-best decoding results
        :rtype: list
        """
        # preprate sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list.index(trans_args.tgt_lang)
        else:
            y = self.sos
        logging.info("<sos> index: " + str(y))
        logging.info("<sos> mark: " + char_list[y])
        logging.info("input lengths: " + str(x.shape[0]))

        h, phone = self.encode(x, phone_sequence)

        h = h.unsqueeze(0)
        phone = phone.unsqueeze(0)

        logging.info("encoder output lengths: " + str(h.size(1)))
        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        if trans_args.maxlenratio == 0:
            maxlen = h.size(1)
        else:
            # maxlen >= 1
            maxlen = max(1, int(trans_args.maxlenratio * h.size(1)))
        minlen = int(trans_args.minlenratio * h.size(1))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        hyp = {"score": 0.0, "yseq": [y]}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            logging.debug("position " + str(i))

            # batchfy
            ys = h.new_zeros((len(hyps), i + 1), dtype=torch.int64)
            for j, hyp in enumerate(hyps):
                ys[j, :] = torch.tensor(hyp["yseq"])
            ys_mask = subsequent_mask(i + 1).unsqueeze(0).to(h.device)

            memory = h.repeat([len(hyps), 1, 1])
            batch_phone = phone.repeat([len(hyps), 1, 1])

            # decode
            if self.is_decoder_use_extra_attn:
                local_scores = self.decoder.forward_one_step(
                    ys, ys_mask, memory, batch_phone,
                )[0]
            else:
                local_scores = self.decoder.forward_one_step(
                    ys, ys_mask, memory,
                )[0]

            hyps_best_kept = []
            for j, hyp in enumerate(hyps):
                local_best_scores, local_best_ids = torch.topk(
                    local_scores[j : j + 1], beam, dim=1
                )

                for j in range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(
                        local_best_scores[0, j]
                    )
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(
                        local_best_ids[0, j]
                    )
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: "
                        + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), trans_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform translation "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            return self.translate(x, phone_sequence, trans_args, char_list)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(
        self, xs_pad, ilens, ys_pad, ys_pad_src, ys_pad_phn, phone_lens
    ):
        """E2E attention calculation.
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_phn: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(
                xs_pad, ilens, ys_pad, ys_pad_src, ys_pad_phn, phone_lens
            )
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention) and m.attn is not None
            ):  # skip MHA for submodules

                ret[name] = m.attn.cpu().numpy()

            if self.is_share_weights:
                for index, attn in enumerate(self.encoder.auxiliary_attns):
                    if attn is not None:
                        ret[
                            f"encoder.auxiliary_encoders.{index}.self_attn"
                        ] = attn.cpu().numpy()

        self.train()
        return ret

    def calculate_all_ctc_probs(
        self, xs_pad, ilens, ys_pad, ys_pad_src, ys_pad_phn, phone_lens
    ):
        """E2E CTC probability calculation.
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_phn: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.asr_weight == 0 or self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(
                xs_pad, ilens, ys_pad, ys_pad_src, ys_pad_phn, phone_lens
            )
        ret = None
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret

    def encode(self, x, phone):
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        phone = torch.as_tensor(phone)

        x_enc_output, _, phone_enc_output, _ = self.encoder(
            x, None, phone, None
        )
        return x_enc_output.squeeze(0), phone_enc_output.squeeze(0)
