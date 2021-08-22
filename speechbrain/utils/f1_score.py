"""Calculate f1 score.

Authors
* Victor Chen 2021
"""
import torch
import torch.nn.functional as F


def micro_f1(pred_onehot, target_onehot):
    """Calculates the micro-f1-score for predicted onehot and target_onehot in a batch.

    Arguments
    ----------
    pred_onehot : tensor
        Predicted log probabilities (batch_size, time, feature).
    target_onehot : tensor
        Target (batch_size, time).
    length : tensor
        Length of target (batch_size,).

    Example
    -------
    >>> onehot = torch.tensor([[1, 0], [0, 1], [1, 0]])
    >>> micro_f1_score = micro_f1(onehot, torch.tensor([[1, 0], [0, 1], [1, 0]]))
    >>> print(micro_f1_score)
    1.0
    """

    tp = torch.sum(pred_onehot * target_onehot)
    # tn = torch.sum((1. - target_onehot) * (1. - pred_onehot))
    fp = torch.sum((1.0 - target_onehot) * pred_onehot)
    fn = torch.sum(target_onehot * (1.0 - pred_onehot))

    epsilon = 1e-7

    numer = tp
    denom = tp + 1 / 2 * (fp + fn + epsilon)
    f1 = numer / denom

    return numer, denom, f1


def macro_f1(pred_onehot, target_onehot):
    """Calculates the macro-f1-score for predicted predicted onehot and target_onehot in a batch.

    Arguments
    ----------
    pred_onehot : tensor
        Predicted predicted onehot (batch_size, time, feature).
    target_onehot : tensor
        Target (batch_size, time).
    length : tensor
        Length of target (batch_size,).

    Example
    -------
    >>> onehot = torch.tensor([[1, 0], [0, 1], [1, 0]])
    >>> macro_f1_score = macro_f1(onehot, torch.tensor([[1, 0], [0, 1], [1, 0]]))
    >>> print(macro_f1_score)
    0.9999998807907104
    """

    tp = torch.sum(pred_onehot * target_onehot, axis=0)
    # tn = torch.sum((1. - target_onehot) * (1. - pred_onehot),axis=0)
    fp = torch.sum((1.0 - target_onehot) * pred_onehot, axis=0)
    fn = torch.sum(target_onehot * (1.0 - pred_onehot), axis=0)

    epsilon = 1e-7

    numers = tp
    denoms = tp + 1 / 2 * (fp + fn + epsilon)
    f1 = torch.mean(numers / denoms)

    return numers, denoms, f1


class F1Stats:
    """Module for calculate the overall one-step-forward prediction f1-scores.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    >>> stats = F1Stats()
    >>> stats.append(probs, torch.tensor([[0], [1], [0]]))
    >>> f1 = stats.summarize()
    >>> print(f1)
    {'micro-f1': 1.0, 'macro-f1': 1.0}
    """

    def __init__(self):
        self.micro_numer = 0
        self.micro_denom = 0
        self.macro_numer = 0
        self.macro_denom = 0

    def append(self, probs, target_indices, length=None):
        """This function is for updating the stats according to the prediction
        and target in the current batch.

        Arguments
        ----------
        pred_onehot : tensor
            Predicted probs (batch_size, time, feature).
        target_onehot : tensor
            Target (batch_size, time).
        length: tensor
            Length of target (batch_size,).
        """

        probs = probs.squeeze()
        n_class = probs.shape[-1]
        pred_indices = probs.argmax(dim=-1)
        pred_onehot = F.one_hot(pred_indices, n_class)

        target_onehot = F.one_hot(target_indices.squeeze(), n_class)

        micro_numer, micro_denom, _ = micro_f1(pred_onehot, target_onehot)
        macro_numers, macro_denoms, _ = macro_f1(pred_onehot, target_onehot)

        self.micro_numer += micro_numer
        self.micro_denom += micro_denom
        self.macro_numer += macro_numers
        self.macro_denom += macro_denoms

    def summarize(self):

        micro_f1 = round(float(self.micro_numer / self.micro_denom), 3)
        macro_f1 = round(
            float(torch.mean(self.macro_numer / self.macro_denom)), 3
        )

        return {"micro-f1": micro_f1, "macro-f1": macro_f1}
