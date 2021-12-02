import math
import julius
from torch import nn
from ..utils import capture_init, center_trim


class BLSTM(nn.Module):
    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            bidirectional=True,
            num_layers=layers,
            hidden_size=dim,
            input_size=dim,
        )
        self.sep_linear = nn.Linear(2 * dim, dim)
        self.recons_linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        sep_code, recons_code = self.sep_linear(x), self.recons_linear(x)
        sep_code = sep_code.permute(1, 2, 0)
        recons_code = recons_code.permute(1, 2, 0)

        return sep_code, recons_code


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Encoder(nn.Module):
    def __init__(
        self,
        activation,
        ch_scale,
        audio_channels,
        channels,
        depth,
        rewrite,
        kernel_size,
        stride,
        growth,
    ):

        super().__init__()
        self.encoder = nn.ModuleList()

        in_channels = audio_channels
        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(in_channels, channels, kernel_size, stride),
                nn.ReLU(),
            ]
            if rewrite:
                encode += [
                    nn.Conv1d(channels, ch_scale * channels, 1),
                    activation,
                ]
            self.encoder.append(nn.Sequential(*encode))

            in_channels = channels
            channels = int(growth * channels)

    def forward(self, x):

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        return x, saved


class Decoder(nn.Module):
    def __init__(
        self,
        sources,
        activation,
        ch_scale,
        audio_channels,
        channels,
        depth,
        rewrite,
        kernel_size,
        stride,
        growth,
        context,
    ):

        super().__init__()

        self.decoder = nn.ModuleList()

        in_channels = audio_channels

        for index in range(depth):
            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(sources) * audio_channels
            if rewrite:
                decode += [
                    nn.Conv1d(channels, ch_scale * channels, context),
                    activation,
                ]
            decode += [
                nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

    def forward(self, x, saved):
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)

        return x


class Generator(nn.Module):
    def __init__(
        self,
        sources,
        activation,
        ch_scale,
        audio_channels,
        channels,
        depth,
        rewrite,
        kernel_size,
        stride,
        growth,
        context,
    ):

        super().__init__()

        self.generator = nn.ModuleList()

        in_channels = audio_channels
        for index in range(depth):
            generate = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(sources) * audio_channels
            if rewrite:
                generate += [
                    nn.Conv1d(channels, ch_scale * channels, context),
                    activation,
                ]
            generate += [
                nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)
            ]
            if index > 0:
                generate.append(nn.ReLU())
            self.generator.insert(0, nn.Sequential(*generate))
            in_channels = channels
            channels = int(growth * channels)

    def forward(self, x, saved):
        for generate in self.generator:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = generate(x)

        return x


class UAE(nn.Module):
    @capture_init
    def __init__(
        self,
        sources,
        audio_channels=2,
        channels=64,
        depth=6,
        rewrite=True,
        glu=True,
        rescale=0.1,
        resample=True,
        kernel_size=8,
        stride=4,
        growth=2.0,
        lstm_layers=2,
        context=3,
        normalize=True,
        samplerate=44100,
        segment_length=4 * 10 * 44100,
    ):
        """
        Args:
            sources (list[str]): list of source names
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            rewrite (bool): add 1x1 convolution to each encoder layer
                and a convolution to each decoder layer.
                For the decoder layer, `context` gives the kernel size.
            glu (bool): use glu instead of ReLU
            resample_input (bool): upsample x2 the input and downsample /2 the output.
            rescale (int): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            lstm_layers (int): number of lstm layers, 0 = no lstm
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time
                steps.
            samplerate (int): stored as meta information for easing
                future evaluations of the model.
            segment_length (int): stored as meta information for easing
                future evaluations of the model. Length of the segments on which
                the model was trained.
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment_length = segment_length

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1

        self.encoder = Encoder(
            activation,
            ch_scale,
            audio_channels,
            channels,
            depth,
            rewrite,
            kernel_size,
            stride,
            growth,
        )
        self.decoder = Decoder(
            sources,
            activation,
            ch_scale,
            audio_channels,
            channels,
            depth,
            rewrite,
            kernel_size,
            stride,
            growth,
            context,
        )
        self.generator = Generator(
            sources,
            activation,
            ch_scale,
            audio_channels,
            channels,
            depth,
            rewrite,
            kernel_size,
            stride,
            growth,
            context,
        )

        in_channels = audio_channels

        for index in range(depth):
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels

        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length when context = 1. If context > 1,
        the two signals can be center trimmed to match.
        For training, extracts should have a valid length.For evaluation
        on full tracks we recommend passing `pad = True` to :method:`forward`.
        """
        if self.resample:
            length *= 2
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix):

        x = mix

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
        else:
            mean = 0
            std = 1

        x = (x - mean) / (1e-5 + std)

        if self.resample:
            x = julius.resample_frac(x, 1, 2)

        # Encoder
        x, saved = self.encoder(x)

        if self.lstm:
            # Code for 2 branch
            sep_code, recons_code = self.lstm(x)

        # Decoder
        sep = self.decoder(sep_code, saved.copy())

        # Generator
        recons = self.generator(recons_code, saved.copy())

        if self.resample:
            sep = julius.resample_frac(sep, 2, 1)
            recons = julius.resample_frac(recons, 2, 1)

        # De-normalize
        sep = sep * std + mean
        sep = sep.view(
            sep.size(0), len(self.sources), self.audio_channels, sep.size(-1)
        )

        recons = recons * std + mean
        recons = recons.view(
            recons.size(0),
            len(self.sources),
            self.audio_channels,
            recons.size(-1),
        )

        return sep, recons
