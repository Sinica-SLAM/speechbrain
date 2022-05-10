"""DCAE implementation in the SpeechBrain style.

Author
-----
* YAO-FEI, CHENG
"""

import logging

from math import ceil
from typing import List, Optional

import torch  # noqa 42

from speechbrain.nnet.activations import Swish
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder

logger = logging.getLogger(__name__)


class PaddedSubsample(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_sizes: List[int],
        strides: List[int],
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.LeakyReLU,
    ):
        super().__init__()
        self.strides = strides
        self.kernels = kernel_sizes
        self.convs = torch.nn.ModuleList()

        for i in range(len(strides)):
            self.convs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=1 if i == 0 else output_size,
                        out_channels=output_size,
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        bias=bias,
                    ),
                    activation(),
                )
            )

        out_channel = input_size
        for kernel, stride in zip(kernel_sizes, strides):
            out_channel = ceil((out_channel - kernel) / stride) + 1

        self.out = torch.nn.Sequential(
            torch.nn.Linear(output_size * out_channel, output_size, bias=bias),
            activation(),
        )

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, time, dimension)
            the input feature
        """

        x = x.unsqueeze(1)

        width_pads = []
        height_pads = []

        for i, conv in enumerate(self.convs):
            original_shape = x.shape
            original_height, original_width = (
                original_shape[-2],
                original_shape[-1],
            )

            width_mod = (original_width - self.kernels[i]) % self.strides[i]
            height_mod = (original_height - self.kernels[i]) % self.strides[i]

            width_pad, height_pad = 0, 0
            if width_mod != 0:
                width_pad = self.strides[i] - width_mod

            if height_mod != 0:
                height_pad = self.strides[i] - height_mod

            width_pads.append(width_pad)
            height_pads.append(height_pad)

            x = torch.nn.functional.pad(x, (width_pad, 0, height_pad, 0))
            x = conv(x)

        batch, channel, time, dim = x.size()
        x = (
            torch.transpose(x, 1, 2)
            .contiguous()
            .view(batch, time, channel * dim)
        )
        x = self.out(x)

        return x, width_pads[::-1], height_pads[::-1]


class PaddedUpSample(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_sizes: List[int],
        strides: List[int],
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.LeakyReLU,
    ):
        super().__init__()
        self.deconvs = torch.nn.ModuleList()
        self.input_size = input_size
        self.factor = 1
        for stride in strides:
            self.factor *= stride

        for i in range(len(strides)):
            self.deconvs.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        in_channels=input_size if i == 0 else output_size,
                        out_channels=1
                        if i == len(strides) - 1
                        else output_size,
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        bias=bias,
                    ),
                    activation(),
                ),
            )

        self.reverse = torch.nn.Sequential(
            torch.nn.Linear(
                input_size, output_size * (input_size // self.factor), bias=bias
            ),
            activation(),
        )

    def forward(self, x, width_pads, height_pads):
        x = self.reverse(x)
        batch, time, _ = x.size()
        x = x.unsqueeze(1).view(batch, self.input_size, time, -1)
        for i, deconv in enumerate(self.deconvs):
            width_pad, height_pad = width_pads[i], height_pads[i]
            x = deconv(x)

            x = x[:, :, : x.shape[-2] - height_pad, : x.shape[-1] - width_pad]

        x = x.squeeze(1)
        return x


class DcaeASR(TransformerASR):
    def __init__(
        self,
        tgt_vocab: int,
        input_size: int,
        kernel_sizes: List[int],
        strides: List[int],
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ffn: int = 2048,
        dropout: float = 0.1,
        activation: torch.nn.Module = torch.nn.ReLU,
        positional_encoding: str = "fixed_abs_sine",
        normalize_before: bool = False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[torch.nn.Module] = Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
    ):
        super().__init__(
            tgt_vocab,
            input_size,
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            d_ffn,
            dropout,
            activation,
            positional_encoding,
            normalize_before,
            kernel_size,
            bias,
            encoder_module,
            conformer_activation,
            attention_type,
            max_length,
            causal,
        )
        self.d_model = d_model
        self.custom_src_module = PaddedSubsample(
            input_size=input_size,
            output_size=d_model,
            kernel_sizes=kernel_sizes,
            strides=strides,
            bias=bias,
            activation=activation,
        )

        self.repr_cnn = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=2 * d_model,
                kernel_size=kernel_sizes[0],
                stride=1,
                padding=kernel_sizes[0] // 2,
            ),
            activation(),
        )

        self.upsampler = PaddedUpSample(
            input_size=2 * d_model,
            output_size=d_model,
            kernel_sizes=kernel_sizes[::-1],
            strides=strides[::-1],
            bias=bias,
            activation=activation,
        )

        if encoder_module == "transformer":
            self.reconstructor = TransformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=causal,
                attention_type=attention_type,
            )
        elif encoder_module == "conformer":
            self.reconstructor = ConformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
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

    def forward(self, src, tgt, wav_len=None, pad_idx=0):
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
        # Downsampe the src vector
        src, width_pads, height_pads = self.custom_src_module(src)

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks(src, tgt, wav_len, pad_idx=pad_idx)

        # add pos encoding to queries if are sinusoidal ones else
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        # Reconstruct the representation to input features
        encoder_out = encoder_out.transpose(1, 2)
        encoder_out = self.repr_cnn(encoder_out)
        encoder_out = encoder_out.transpose(1, 2)

        reconstructed_src = self.upsampler(encoder_out, width_pads, height_pads)

        # Re-calculate padding mask after being upsampled
        rec_src_key_padding_mask = None
        rec_src_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * reconstructed_src.shape[1])
            rec_src_key_padding_mask = (
                torch.arange(reconstructed_src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )

        reconstructed_src, _ = self.reconstructor(
            src=reconstructed_src,
            src_mask=rec_src_mask,
            src_key_padding_mask=rec_src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        tgt = self.custom_tgt_module(tgt)

        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            src = src + self.positional_encoding_decoder(src)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None

        encoder_out = encoder_out[:, :, self.d_model :]
        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )

        return encoder_out, reconstructed_src, decoder_out
