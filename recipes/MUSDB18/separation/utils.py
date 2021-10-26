# import errno
# import functools
# import hashlib
# import inspect
# import io
# import os
# import random
# import socket
# import tempfile
# import warnings
# import zlib
# from contextlib import contextmanager

# import torch as th
# import tqdm
# from torch import distributed
from torch.nn import functional as F

# from torch import nn


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


# def process_chunk(separate, mix, segment=44100 * 10 * 4, sources=4, samplerate=44100, split=True,
#                 overlap=0.25, transition_power=1., progress=False):
#     """
#     Apply model to a given mixture.
#     Args:
#         shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
#             and apply the oppositve shift to the output. This is repeated `shifts` time and
#             all predictions are averaged. This effectively makes the model time equivariant
#             and improves SDR by up to 0.2 points.
#         split (bool): if True, the input will be broken down in 8 seconds extracts
#             and predictions will be performed individually on each and concatenated.
#             Useful for model with large memory footprint like Tasnet.
#         progress (bool): if True, show a progress bar (requires split=True)
#     """
#     assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
#     device = mix.device
#     channels, length = mix.shape
#     if split:
#         out = th.zeros(sources, channels, length, device=device)
#         sum_weight = th.zeros(length, device=device)
#         stride = int((1 - overlap) * segment)
#         offsets = range(0, length, stride)
#         scale = stride / samplerate
#         if progress:
#             offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
#         # We start from a triangle shaped weight, with maximal weight in the middle
#         # of the segment. Then we normalize and take to the power `transition_power`.
#         # Large values of transition power will lead to sharper transitions.
#         weight = th.cat([th.arange(1, segment // 2 + 1),
#                          th.arange(segment - segment // 2, 0, -1)]).to(device)
#         assert len(weight) == segment
#         # If the overlap < 50%, this will translate to linear transition when
#         # transition_power is 1.
#         weight = (weight / weight.max())**transition_power
#         for offset in offsets:
#             chunk = TensorChunk(mix, offset, segment)
#             chunk_out = process_chunk(separate, chunk)
#             chunk_length = chunk_out.shape[-1]
#             out[..., offset:offset + segment] += weight[:chunk_length] * chunk_out
#             sum_weight[offset:offset + segment] += weight[:chunk_length]
#             offset += segment
#         assert sum_weight.min() > 0
#         out /= sum_weight
#         return out

#     else:
#         with th.no_grad():
#             out = separate(mix.unsqueeze(0))[0]
#         return center_trim(out, length)

# if __name__ == '__main__':

#     x = th.rand(4,100)
#     model = nn.Linear(100,200)
#     pred = apply_model(model, x)

#     # output = model(x)
#     print(pred.shape)
