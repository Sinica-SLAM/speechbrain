"""Gradient based subword tokenization in the SpeechBrain sytle.

Authors
* YAO FEI, CHENG 2021
"""

from typing import Optional
import torch
import logging
import torch.nn.functional as F

from torch import nn

from speechbrain.utils.scatter_mean import scatter_mean

logger = logging.getLogger(__name__)


class DepthwiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            dim_in, dim_out, kernel_size, groups=dim_in, bias=bias
        )
        self.proj_out = nn.Conv1d(dim_out, dim_out, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return self.proj_out(x)


class GBST(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        dictionary_size: int,
        blocks_size: int,
        pad_index: int = 0,
        global_scores_weight: float = 0.0,
    ):
        super().__init__()

        self.character_embedding = nn.Embedding(
            dictionary_size, embedding_size, padding_idx=pad_index
        )
        self.score_function = nn.Linear(embedding_size, 1)
        self.pos_conv = DepthwiseConv1d(
            embedding_size, embedding_size, blocks_size
        )

        self.blocks_size = blocks_size
        self.pad_index = pad_index
        self.global_scores_weight = global_scores_weight
        if self.global_scores_weight > 0:
            self.global_score_function = nn.Sequential(
                nn.Linear(self.blocks_size, self.blocks_size), nn.ReLU(),
            )

    def forward(
        self,
        sequence: torch.LongTensor,
        group_id: torch.LongTensor,
        global_scores: Optional[torch.FloatTensor] = None,
    ):
        # Calculate character embedding
        embed_sequence = self.character_embedding(sequence)
        blocks_mask = torch.cat((sequence.unsqueeze(1), group_id), dim=1)

        # Calculate positions for each token
        embed_sequence = F.pad(
            embed_sequence,
            pad=(0, 0, 0, self.blocks_size - 1),
            value=self.pad_index,
        )

        embed_sequence = embed_sequence.permute(0, 2, 1)
        embed_sequence = self.pos_conv(embed_sequence)
        embed_sequence = embed_sequence.permute(0, 2, 1)

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
            embed_sequence_in_layer = embed_sequence.detach().clone()

            # Mask out zeros since they are not the vocabs
            sequence_in_layer_mask = (
                group_id_in_layer.unsqueeze(-1)
                .eq(self.pad_index)
                .to(sequence.device)
            )
            embed_sequence_in_layer = embed_sequence_in_layer.masked_fill(
                sequence_in_layer_mask, 0
            )

            # Since scatter mean will index from 0
            # So, shift index by 1
            for batch in range(len(pad_group_id)):
                # Replac the pad index to group id
                group_id_in_layer_in_batch = group_id_in_layer[batch]
                group_id_in_layer_in_batch[
                    group_id_in_layer_in_batch == self.pad_index
                ] = pad_group_id[batch]
                group_id_in_layer_in_batch = group_id_in_layer_in_batch - 1
                group_id_in_layer[batch] = group_id_in_layer_in_batch

            blocked_sequence = scatter_mean(
                embed_sequence_in_layer, group_id_in_layer, dim=1
            )

            # Calculate frequencies for each group
            # As mention above, the original sequence was subsampled
            # Upsample them by the length of vocabs
            # Since bincount doesn't support batch-wise operation
            # So, loop over the batch
            block_representation = []
            for batch in range(len(group_id_in_layer)):
                # Calculate frequencies based on the length of vocabs
                group_frequency = torch.bincount(group_id_in_layer[batch])
                block_sequence_length = blocked_sequence.shape[1]
                pad_length = block_sequence_length - group_frequency.shape[0]
                # Pad upsampled sequences to original lengths
                group_frequency = F.pad(
                    group_frequency, pad=(0, pad_length), value=self.pad_index
                )
                block_sequence_in_batch = torch.repeat_interleave(
                    blocked_sequence[batch], group_frequency, dim=0
                )
                block_representation.append(block_sequence_in_batch)

            block_representation = torch.stack(block_representation)
            block_representations.append(block_representation)

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

        # Fuse the global scores with weight
        if self.global_scores_weight > 0:
            global_scores = global_scores.permute(0, 2, 1)
            global_scores_mask = global_scores.eq(self.pad_index).to(
                sequence.device
            )
            global_scores = self.global_score_function(global_scores)
            global_scores = global_scores.masked_fill(
                global_scores_mask, max_neg_value
            )
            global_scores = global_scores.softmax(dim=2)
            global_scores = global_scores.unsqueeze(-1)
            scores = (
                scores * (1 - self.global_scores_weight)
                + self.global_scores_weight * global_scores
            )

        # Weighted sum over block representations
        embed_sequence = torch.mul(block_representations, scores).sum(dim=2)

        return embed_sequence
