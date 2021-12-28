import random
import torch
from torch import nn
from julius import ResampleFrac
from torchaudio.transforms import TimeStretch


class Spec(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x, inverse=False):
        shape = x.shape
        if inverse:
            spec = x.reshape(-1, *shape[-2:])
            return torch.istft(
                spec, self.n_fft, self.hop_length, self.n_fft, self.window,
            ).view(*shape[:-2], -1)
        x = x.reshape(-1, x.shape[-1])
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            self.window,
            return_complex=True,
        )
        # (batch, channels, nfft // 2 + 1, frames)
        return spec.view(*shape[:-1], *spec.shape[-2:])


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """

    def __init__(self, shift=8192):
        super().__init__()
        self.shift = shift

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        if self.shift > 0 and length > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                offsets = torch.randint(
                    self.shift, [batch, sources, 1, 1], device=wav.device
                )
                offsets = offsets.expand(-1, -1, channels, -1)
                indexes = torch.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    """

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = torch.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1, -1, -1, time)
            right = 1 - left
            wav = torch.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    """

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = torch.randint(
                2,
                (batch, sources, 1, 1),
                device=wav.device,
                dtype=torch.float32,
            )
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """

    def __init__(self, group_size=4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separatly. This allow to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device

        if self.training:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(
                    f"Batch size {batch} must be divisible by group size {group_size}"
                )
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = torch.argsort(
                torch.rand(groups, group_size, streams, 1, 1, device=device),
                dim=1,
            )
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1.0, min=0.25, max=1.25):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = torch.empty(batch, streams, 1, 1, device=device).uniform_(
                self.min, self.max
            )
            wav *= scales
        return wav


class _DeviceTransformBase(nn.Module):
    def __init__(self, rand_size, p=0.2):
        super().__init__()
        self.p = p
        self.rand_size = rand_size

    def _transform(self, stems, index):
        raise NotImplementedError

    def forward(self, stems: torch.Tensor):
        """
        Args:
            stems (torch.Tensor): (B, Num_sources, Num_channels, L)
        Return:
            perturbed_stems (torch.Tensor): (B, Num_sources, Num_channels, L')
        """
        shape = stems.shape
        orig_len = shape[-1]
        stems = stems.reshape(-1, *shape[-2:])
        select_mask = torch.rand(stems.shape[0], device=stems.device) < self.p
        if not torch.any(select_mask):
            return stems.view(*shape)

        select_idx = torch.where(select_mask)[0]
        perturbed_stems = torch.zeros_like(stems)
        perturbed_stems[~select_mask] = stems[~select_mask]
        selected_stems = stems[select_mask]
        rand_idx = torch.randint(
            self.rand_size, (selected_stems.shape[0],), device=stems.device
        )

        for i in range(self.rand_size):
            mask = rand_idx == i
            if not torch.any(mask):
                continue
            masked_stems = selected_stems[mask]
            perturbed_audio = self._transform(masked_stems, i)

            diff = perturbed_audio.shape[-1] - orig_len

            put_idx = select_idx[mask]
            if diff >= 0:
                perturbed_stems[put_idx] = perturbed_audio[..., :orig_len]
            else:
                perturbed_stems[put_idx, :, : orig_len + diff] = perturbed_audio

        perturbed_stems = perturbed_stems.view(*shape)
        return perturbed_stems


class SpeedPerturb(_DeviceTransformBase):
    def __init__(self, orig_freq=44100, speeds=[90, 100, 110], **kwargs):
        super().__init__(len(speeds), **kwargs)
        self.orig_freq = orig_freq
        self.resamplers = nn.ModuleList()
        self.speeds = speeds
        for s in self.speeds:
            new_freq = self.orig_freq * s // 100
            self.resamplers.append(ResampleFrac(self.orig_freq, new_freq))

    def _transform(self, stems, index):
        y = self.resamplers[index](stems.view(-1, stems.shape[-1])).view(
            *stems.shape[:-1], -1
        )
        return y


class RandomPitch(_DeviceTransformBase):
    def __init__(
        self, semitones=[-2, -1, 0, 1, 2], n_fft=2048, hop_length=512, **kwargs
    ):
        super().__init__(len(semitones), **kwargs)
        self.resamplers = nn.ModuleList()

        semitones = torch.tensor(semitones, dtype=torch.float32)
        rates = 2 ** (-semitones / 12)
        rrates = rates.reciprocal()
        rrates = (rrates * 100).long()
        rrates[rrates % 2 == 1] += 1
        rates = 100 / rrates

        self.register_buffer("rates", rates)
        self.spec = Spec(n_fft, hop_length)
        self.stretcher = TimeStretch(hop_length, n_freq=n_fft // 2 + 1)

        for rr in rrates.tolist():
            self.resamplers.append(ResampleFrac(rr, 100))

    def _transform(self, stems, index):
        spec = torch.view_as_real(self.spec(stems))
        stretched_spec = self.stretcher(spec, self.rates[index])
        stretched_stems = self.spec(
            torch.view_as_complex(stretched_spec), inverse=True
        )
        shifted_stems = self.resamplers[index](
            stretched_stems.view(-1, stretched_stems.shape[-1])
        ).view(*stretched_stems.shape[:-1], -1)
        return shifted_stems


if __name__ == "__main__":
    trsfn = nn.Sequential(
        RandomPitch(),
        SpeedPerturb(),
        Shift(1 * 44100),
        FlipSign(),
        FlipChannels(),
        Scale(),
        Remix(group_size=1),
    )

    # x = torch.randn(3, 4, 2, 22050)
    x = torch.randn(3, 4, 2, 441000)
    y = trsfn(x)
    print(y.shape)
    # print(y.shape, x[0, 0, 0, -100:], y[0, 0, 0, -100:])
