"""Gradient based subword tokenization in the SpeechBrain sytle.

Authors
* YAO FEI, CHENG 2021
"""

import torch
import logging
import torch.nn.functional as F

from torch import nn

from speechbrain.utils.scatter_mean import scatter_mean

logger = logging.getLogger(__name__)


class GBST(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        dictionary_size: int,
        blocks_size: int,
        pad_index: int = 0,
    ):
        super().__init__()

        self.character_embedding = nn.Embedding(
            dictionary_size, embedding_size, padding_idx=pad_index
        )
        self.score_function = nn.Linear(embedding_size, 1)

        self.blocks_size = blocks_size
        self.pad_index = pad_index

    def forward(
        self, sequence: torch.LongTensor, group_id: torch.LongTensor,
    ):
        # Calculate character embedding
        embed_sequence = self.character_embedding(sequence)
        blocks_mask = torch.cat((sequence.unsqueeze(1), group_id), dim=1)

        # Calculate the index frequency
        pad_group_ids, _ = torch.max(group_id, dim=2)
        pad_group_ids = pad_group_ids + 1

        # The first layer is unigram, and the following layers are possible segmentations
        block_representations = [embed_sequence]

        # Calculate the block representation within the layer
        for layer in range(self.blocks_size - 1):
            group_id_in_layer = group_id[:, layer]
            pad_group_id = pad_group_ids[:, layer]

            # Calculate mean for each vocab
            # it returns [batch, longest_length_among_the_batch, dimension]
            # original sequence lengths will be subsampled to longest_length_among_the_batch
            blocked_sequence = scatter_mean(
                embed_sequence.detach().clone(), group_id_in_layer, dim=1
            )

            group_frequencies = []
            # Calculate frequencies for each group
            # As mention above, the original sequence was subsampled
            # Upsample them by the length of vocabs
            for batch in range(len(pad_group_id)):
                # Replac the pad index to group id
                group_id_in_layer_in_batch = group_id_in_layer[batch]
                group_id_in_layer_in_batch[
                    group_id_in_layer_in_batch == self.pad_index
                ] = pad_group_id[batch]
                group_id_in_layer_in_batch = group_id_in_layer_in_batch - 1

                # Calculate frequencies based on the length of vocabs
                group_frequency = torch.bincount(group_id_in_layer_in_batch)
                block_sequence_length = blocked_sequence.shape[1]
                pad_length = block_sequence_length - group_frequency.shape[0]
                # Pad upsampled sequences to original lengths
                group_frequency = F.pad(
                    group_frequency, pad=(0, pad_length), value=self.pad_index
                )
                group_frequencies.append(group_frequency)

            # Batch-wise group frequencies
            group_frequencies = torch.stack(group_frequencies)

            # Upsample based on frequency
            blocked_sequence = torch.repeat_interleave(
                blocked_sequence, group_frequency, dim=1
            )
            block_representations.append(blocked_sequence)

        block_representations = torch.stack(block_representations, dim=2)

        # Calculate scores
        scores = self.score_function(block_representations)
        scores = scores.squeeze(-1)

        # Calculate sequence masks
        blocks_mask = (
            blocks_mask.permute(0, 2, 1).eq(self.pad_index).to(sequence.device)
        )

        # Maske out padded tokens
        max_neg_value = -torch.finfo(scores.dtype).max
        scores = scores.masked_fill(blocks_mask, max_neg_value)
        scores = scores.softmax(dim=2)
        scores = scores.unsqueeze(-1)

        # Weighted sum over block representations
        embed_sequence = torch.mul(block_representations, scores).sum(dim=2)

        return embed_sequence
