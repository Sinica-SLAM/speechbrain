import functools
import logging
import math
import torch
from torch.nn import functional as F
import random
import tqdm

logger = logging.getLogger(__name__)


def capture_init(init):
    """
    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def center_trim(tensor, reference):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError(
            "tensor must be larger than reference. " f"Delta is {delta}."
        )
    if delta:
        tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
    return tensor


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    # gcd=Greatest Common Divisor
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )
    frame = frame.clone().detach().long().to(signal.device)
    # frame = signal.new_tensor(frame).clone().long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(
        *outer_dimensions, output_subframes, subframe_length
    )
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        self.tensor = tensor
        self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(
            self.tensor[..., correct_start:correct_end], (pad_left, pad_right)
        )
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, torch.Tensor)
        return TensorChunk(tensor_or_chunk)


def apply_model(
    model,
    mix,
    shifts=None,
    split=False,
    overlap=0.25,
    transition_power=1.0,
    progress=False,
    no_pad=False,
):
    """
    Apply model to a given mixture.
    Args:
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down into chunks with segment_length
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        progress (bool): if True, show a progress bar (requires split=True)
    """
    assert (
        transition_power >= 1
    ), "transition_power < 1 leads to weird behavior."
    device = mix.device
    _, channels, length = mix.shape  # (B, C, T)

    if split:
        out = torch.zeros(len(model.sources), channels, length, device=device)
        sum_weight = torch.zeros(length, device=device)
        segment = model.segment_length
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        scale = stride / model.samplerate
        if progress:
            offsets = tqdm.tqdm(
                offsets, unit_scale=scale, ncols=120, unit="seconds"
            )
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = torch.cat(
            [
                torch.arange(1, segment // 2 + 1),
                torch.arange(segment - segment // 2, 0, -1),
            ]
        ).to(device)
        assert len(weight) == segment
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max()) ** transition_power
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            chunk_out = apply_model(model, chunk, shifts=shifts, no_pad=no_pad)
            chunk_length = chunk_out.shape[-1]
            out[..., offset : offset + segment] += (
                weight[:chunk_length] * chunk_out
            )
            sum_weight[offset : offset + segment] += weight[:chunk_length]
            offset += segment
        assert sum_weight.min() > 0
        out /= sum_weight
        return out
    elif shifts:
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(
                padded_mix, offset, length + max_shift - offset
            )
            shifted_out = apply_model(model, shifted, no_pad=no_pad)
            out += shifted_out[..., max_shift - offset :]
        out /= shifts
        return out

    elif no_pad:
        mix = tensor_chunk(mix)
        mix = mix.padded(length)
        with torch.no_grad():
            out = model(mix)[-1]
        out = out.permute(0, 3, 2, 1)[0]  # (N, C, T)
        return out

    else:
        valid_length = model.valid_length(length)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length)
        with torch.no_grad():
            out = model(padded_mix)[0]
        return center_trim(out, length)
