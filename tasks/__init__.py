import abc
from typing import TypedDict

import torch
import torch.nn.functional as F
import numpy as np

Batch = TypedDict("Batch", {"input": torch.Tensor, "output": torch.Tensor})


class GeneralizationTask(abc.ABC):
    """A task for the generalization project.

    Exposes a sample_batch method, and some details about input/output sizes,
    losses and accuracies.
    """

    @abc.abstractmethod
    def sample_batch(
        self, rng: np.random.Generator, batch_size: int, length: int
    ) -> Batch:
        """Returns a batch of inputs/outputs."""

    def pointwise_loss_fn(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Returns the pointwise loss between an output and a target."""
        return -target * F.log_softmax(output, dim=-1)

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns the accuracy between an output and a target."""
        return (output.argmax(dim=-1) == target.argmax(dim=-1)).float()

    def accuracy_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Returns a mask to compute the accuracies, to remove the superfluous ones."""
        # Target is of shape (B, T, C) where C is the number of classes.
        # We want a mask per input (B, T), so we take this shape.
        return torch.ones(target.shape[:-1])

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        """Returns the size of the input of the models trained on this task."""

    @property
    @abc.abstractmethod
    def output_size(self) -> int:
        """Returns the size of the output of the models trained on this task."""

    def output_length(self, input_length: int) -> int:
        """Returns the length of the output, given an input length."""
        del input_length
        return 1
