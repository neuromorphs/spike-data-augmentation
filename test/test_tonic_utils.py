from typing import Any, Tuple

import pytest
import tonic.transforms as transforms
import torch
from tonic.collation import PadTensors
from utils import create_random_input


class DummyDataset:
    """Simulates event-based sensor data with transformations."""

    def __init__(self, events: Tuple[Any, ...], transform: transforms.Compose):
        self.events = events
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Applies transformation to an event and returns it with a dummy label."""
        return self.transform(self.events[index]), 1

    def __len__(self) -> int:
        """Returns the number of events in the dataset."""
        return len(self.events)


def test_pytorch_batch_collation_dense_tensor():
    """Tests collation of event data into dense tensors using a PyTorch DataLoader."""
    events1, sensor_size = create_random_input()
    events2, _ = create_random_input()

    time_window = 1000
    transform = transforms.Compose(
        [transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)]
    )
    dataset = DummyDataset((events1[:5000], events2), transform)
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=PadTensors(batch_first=False),
        batch_size=batch_size,
    )

    batch, label = next(iter(dataloader))

    max_time = int(events2["t"][-1])
    assert batch.shape[0] == max_time // time_window
    assert batch.shape[1] == batch_size
    assert batch.shape[2] == sensor_size[2]
