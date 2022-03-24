"""metrics.py

Metrics Utilities.
The Module provides useful metrics as
function and PyTorch Module APIs.
"""
from torch import Tensor
from torch.nn import Module


def accuracy(logits: Tensor, labels: Tensor, dim: int = -1) -> Tensor:
    """Accuracy
    
    Parameters
    ----------
    logits, labels: Tensor
        logits and labels to compute the accuracy respectively
        of size (num_samples, num_classes) and (num_samples, )
    dim: int (default: -1)
        dimension to which to apply the argmax

    Returns
    -------
    accuracy: Tensor
        mean batch accuracy between 0 and 1
    """
    return (logits.argmax(dim=dim) == labels).sum() / logits.size(0)


class Accuracy(Module):
    """Accuracy"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: Tensor, labels: Tensor, dim: int = -1) -> Tensor:
        """Forward
        
        Parameters
        ----------
        logits, labels: Tensor
            logits and labels to compute the accuracy respectively
            of size (num_samples, num_classes) and (num_samples, )
        dim: int (default: -1)
            dimension to which to apply the argmax

        Returns
        -------
        accuracy: Tensor
            mean batch accuracy between 0 and 1
        """
        return accuracy(logits, labels, dim=dim)


class RunningAverage:
    """Running Average"""

    def __init__(self) -> None:
        self.current = 0
        self.value = 0.

    def update(self, value: float) -> None:
        """Update
        
        Update average:
            `old <- (old * t + new) / (t + 1)`

        Parameters
        ----------
        value: float
            new value
        """
        self.value = self.value * self.current + value
        self.current += 1
        self.value = self.value / self.current