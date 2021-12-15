"""AlloST in the SpeechBrain sytle.

Authors
* YAO FEI, CHENG 2021
"""

import torch  # noqa 42
import logging

from torch import nn
from typing import Optional

import speechbrain as sb

from speechbrain.nnet.activations import Swish
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from speechbrain.nnet.attention import (
    RelPosMHAXL,
    MultiheadAttention,
    PositionalwiseFeedForward,
)
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerDecoderLayer,
    TransformerDecoder,
    TransformerEncoder,
    get_key_padding_mask,
    get_lookahead_mask,
)


logger = logging.getLogger(__name__)


class AlloSTDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        attention_type="regularMHA",
        causal=False,
        fusion_type="vanilla",
    ):
        super().__init__(
            d_ffn,
            nhead,
            d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            attention_type=attention_type,
            causal=causal,
        )
        self.nhead = nhead
        self.fusion_type = fusion_type

        if attention_type == "regularMHA":
            self.self_attn = MultiheadAttention(
                nhead=nhead, d_model=d_model, kdim=kdim, vdim=vdim, dropout=0,
            )
            self.mutihead_attn = MultiheadAttention(
                nhead=nhead, d_model=d_model, kdim=kdim, vdim=vdim, dropout=0,
            )
            if fusion_type == "stacked":
                self.auxiliary_attn = MultiheadAttention(
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=0,
                )
        elif attention_type == "RelPosMHAXL":
            self.self_attn = RelPosMHAXL(
                d_model, nhead, 0, mask_pos_future=causal
            )
            self.mutihead_attn = RelPosMHAXL(
                d_model, nhead, 0, mask_pos_future=causal
            )
            if fusion_type == "stacked":
                self.auxiliary_attn = RelPosMHAXL(
                    d_model, nhead, 0, mask_pos_future=causal
                )

        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        # normalization layers
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.norm3 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

        if fusion_type == "stacked":
            self.norm4 = LayerNorm(d_model, eps=1e-6)
            self.dropout4 = torch.nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        phone_memory,
        tgt_mask=None,
        memory_mask=None,
        phone_memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        phone_memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
        pos_embs_phone=None,
    ):
        """
        Arguments
        ----------
        tgt: tensor
            The sequence to the decoder layer (required).
        memory: tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask: tensor
            The mask for the tgt sequence (optional).
        memory_mask: tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask: tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask: tensor
            The mask for the memory keys per batch (optional).
        """
        if self.normalize_before:
            tgt1 = self.norm1(tgt)
        else:
            tgt1 = tgt

        # self-attention over the target sequence
        tgt2, self_attn = self.self_attn(
            query=tgt1,
            key=tgt1,
            value=tgt1,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            pos_embs=pos_embs_tgt,
        )

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        if self.normalize_before:
            tgt1 = self.norm2(tgt)
        else:
            tgt1 = tgt

        # multi-head attention over the target sequence and encoder states

        tgt2, multihead_attention = self.mutihead_attn(
            query=tgt1,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            pos_embs=pos_embs_src,
        )

        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if self.fusion_type == "stacked":
            if self.normalize_before:
                tgt1 = self.norm4(tgt)
            else:
                tgt1 = tgt

            tgt2, multihead_attention = self.auxiliary_attn(
                query=tgt1,
                key=phone_memory,
                value=phone_memory,
                attn_mask=phone_memory_mask,
                key_padding_mask=phone_memory_key_padding_mask,
                pos_embs=pos_embs_phone,
            )

            # add & norm
            tgt = tgt + self.dropout4(tgt2)
            if not self.normalize_before:
                tgt = self.norm4(tgt)

        if self.normalize_before:
            tgt1 = self.norm3(tgt)
        else:
            tgt1 = tgt

        tgt2 = self.pos_ffn(tgt1)

        # add & norm
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt, self_attn, multihead_attention


class AlloSTDecoder(TransformerDecoder):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        attention_type="regularMHA",
        causal=False,
        fusion_type="vanilla",
    ):
        super().__init__(
            num_layers,
            nhead,
            d_ffn,
            d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            causal=causal,
            attention_type=attention_type,
        )

        self.layers = torch.nn.ModuleList(
            [
                AlloSTDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    attention_type=attention_type,
                    causal=causal,
                    fusion_type=fusion_type,
                )
            ]
        )

    def forward(
        self,
        tgt,
        memory,
        phone_memory,
        tgt_mask=None,
        memory_mask=None,
        phone_memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        phone_memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
        pos_embs_phone=None,
    ):
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                phone_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                phone_memory_mask=phone_memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                phone_memory_key_padding_mask=phone_memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
                pos_embs_phone=pos_embs_phone,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)

        return output, self_attns, multihead_attns


class AlloST(TransformerASR):
    """This is an implementation of transformer model for ST.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    tgt_vocab: int
        Size of vocabulary.
    input_size: int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
        (default=512).
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int, optional
        The number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers : int, optional
        The number of sub-decoder-layers in the decoder (default=6).
    dim_ffn : int, optional
        The dimension of the feedforward network model (default=2048).
    dropout : int, optional
        The dropout value (default=0.1).
    activation : torch.nn.Module, optional
        The activation function of FFN layers.
        Recommended: relu or gelu (default=relu).
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        Choose between Conformer and Transformer for the encoder. The decoder is fixed to be a Transformer.
    conformer_activation: torch.nn.Module, optional
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    ctc_weight: float
        The weight of ctc for asr task
    asr_weight: float
        The weight of asr task for calculating loss
    mt_weight: float
        The weight of mt task for calculating loss
    asr_tgt_vocab: int
        The size of the asr target language
    mt_src_vocab: int
        The size of the mt source language
    Example
    -------

    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_auxiliary_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        auxiliary_encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        is_encoder_fusion: Optional[bool] = False,
        decoder_fusion_type: Optional[str] = "vanilla",
        custom_phone_module: Optional[nn.Module] = None,
        pre_trained_size: Optional[int] = 768,
    ):
        super().__init__(
            tgt_vocab=tgt_vocab,
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
        )
        self.is_encoder_fusion = is_encoder_fusion
        self.custom_phone_module = custom_phone_module
        self.normalize_before = normalize_before
        self.auxiliary_encoder_module = auxiliary_encoder_module

        if auxiliary_encoder_module == "transformer":
            self.auxiliary_encoder = TransformerEncoder(
                nhead=nhead,
                num_layers=num_auxiliary_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=self.causal,
                attention_type=self.attention_type,
            )
        elif auxiliary_encoder_module == "conformer":
            self.auxiliary_encoder = ConformerEncoder(
                nhead=nhead,
                num_layers=num_auxiliary_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=conformer_activation,
                kernel_size=kernel_size,
                bias=bias,
                causal=self.causal,
                attention_type=self.attention_type,
            )
            assert (
                normalize_before
            ), "normalize_before must be True for Conformer"

            assert (
                conformer_activation is not None
            ), "conformer_activation must not be None"
            if is_encoder_fusion:
                if attention_type == "regularMHA":
                    self.fusion_attn = MultiheadAttention(nhead, d_model, 0,)
                else:
                    self.fusion_attn = RelPosMHAXL(
                        d_model, nhead, 0, mask_pos_future=causal,
                    )
                self.fusion_norm = sb.nnet.normalization.LayerNorm(
                    d_model, eps=1e-6
                )
            self.fusion_dropout = torch.nn.Dropout(p=dropout)
        elif auxiliary_encoder_module == "pre-trained":
            self.pre_trained_linear = torch.nn.Linear(pre_trained_size, d_model)

        self.decoder = AlloSTDecoder(
            num_layers=num_decoder_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            causal=True,
            attention_type="regularMHA",  # always use regular attention in decoder
            fusion_type=decoder_fusion_type,
        )

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(
        self, src, src_phone, tgt, wav_len, pad_idx=0,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """
        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks(src, tgt, wav_len, pad_idx=pad_idx)

        src_phone_key_padding_mask = get_key_padding_mask(
            src_phone, pad_idx=pad_idx
        )
        src_phone_mask = None

        src = self.custom_src_module(src)
        if self.custom_phone_module is not None:
            src_phone = self.custom_phone_module(src_phone)

        # add pos encoding to queries if are sinusoidal ones else
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
            pos_embs_phone = self.positional_encoding(src_phone)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            src_phone = src_phone + self.positional_encoding(src_phone)
            pos_embs_encoder = None
            pos_embs_phone = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        if self.auxiliary_encoder_module == "pre-trained":
            phone_encoder_out = self.pre_trained_linear(src_phone)
        else:
            phone_encoder_out, _ = self.auxiliary_encoder(
                src=src_phone,
                src_mask=src_phone_mask,
                src_key_padding_mask=src_phone_key_padding_mask,
                pos_embs=pos_embs_phone,
            )

        if self.is_encoder_fusion:
            if self.normalize_before:
                encoder_out1 = self.fusion_norm(encoder_out)
            else:
                encoder_out1 = encoder_out

            encoder_out, _ = self.fusion_attn(
                query=encoder_out1,
                key=phone_encoder_out,
                value=phone_encoder_out,
                attn_mask=src_phone_mask,
                key_padding_mask=src_phone_key_padding_mask,
                pos_embs=pos_embs_phone,
            )

            # add & norm
            encoder_out = encoder_out + self.fusion_dropout(encoder_out)
            if not self.normalize_before:
                encoder_out = self.fusion_norm(encoder_out)

        tgt = self.custom_tgt_module(tgt)

        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            src = src + self.positional_encoding_decoder(src)
            src_phone = src_phone + self.positional_encoding_decoder(src_phone)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
            pos_embs_phone = None
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None
            pos_embs_phone = None

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            phone_memory=phone_encoder_out,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            phone_memory_mask=src_phone_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            phone_memory_key_padding_mask=src_phone_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
            pos_embs_phone=pos_embs_phone,
        )

        return encoder_out, phone_encoder_out, decoder_out

    def decode(self, tgt, encoder_out, phone_representation):
        tgt_mask = get_lookahead_mask(tgt)
        tgt = self.custom_tgt_module(tgt)
        if self.attention_type == "RelPosMHAXL":
            # we use fixed positional encodings in the decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            encoder_out = encoder_out + self.positional_encoding_decoder(
                encoder_out
            )
            phone_representation = (
                phone_representation
                + self.positional_encoding_decoder(phone_representation)
            )
            # pos_embs_target = self.positional_encoding(tgt)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
            pos_embs_phone_encoder = None
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
            pos_embs_target = None
            pos_embs_encoder = None
            pos_embs_phone_encoder = None

        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            phone_representation,
            tgt_mask=tgt_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
            pos_embs_phone=pos_embs_phone_encoder,
        )
        return prediction, multihead_attns[-1]
