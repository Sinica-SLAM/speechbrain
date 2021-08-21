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

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * precision * recall / (precision + recall + epsilon)

    return float(f1)


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

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * precision * recall / (precision + recall + epsilon)

    return float(torch.mean(f1))


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
        self.micro_f1 = 0
        self.macro_f1 = 0

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

        micro_f1_score = micro_f1(pred_onehot, target_onehot)
        macro_f1_score = macro_f1(pred_onehot, target_onehot)

        self.micro_f1 = micro_f1_score
        self.macro_f1 = macro_f1_score

    def summarize(self):
        return {
            "micro-f1": round(self.micro_f1, 3),
            "macro-f1": round(self.macro_f1, 3),
        }
