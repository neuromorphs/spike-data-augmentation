import numpy as np
from tonic.slicers import (SliceAtIndices, SliceAtTimePoints,
                           SliceByEventCount, SliceByTime,
                           slice_events_at_indices)


def test_slice_at_indices_class() -> None:
    """Test slicing at specific indices using the SliceAtIndices class."""
    slicer = SliceAtIndices(start_indices=[0, 3, 5], end_indices=[3, 5, 7])
    data = np.arange(7)
    slices, _ = slicer.slice(data, None)
    assert len(slices) == 3


def test_slice_at_indices_method() -> None:
    """Test slicing at specific indices using the slice_events_at_indices function."""
    data = np.arange(7)
    slices = slice_events_at_indices(
        data, start_indices=[0, 3, 5], end_indices=[3, 5, 7]
    )
    assert len(slices) == 3


def test_slice_at_time_points_class() -> None:
    """Test slicing at specific time points using the SliceAtTimePoints class."""
    data = np.arange(100).astype(dtype=[("t", float)])
    slicer = SliceAtTimePoints(start_tw=[10, 20, 50], end_tw=[20, 50, 100])
    slices, _ = slicer.slice(data=data, targets=None)
    assert len(slices) == 3


def test_slice_by_event_count() -> None:
    """Test slicing by event count using the SliceByEventCount class."""
    data = np.arange(100)
    slicer = SliceByEventCount(event_count=50, overlap=25, include_incomplete=False)
    data_slice, _ = slicer.slice(data, None)
    assert len(data_slice) == 3


def test_slice_by_time() -> None:
    """Test slicing by time using the SliceByTime class."""
    data = np.arange(101).astype(dtype=[("t", float)])
    slicer = SliceByTime(time_window=5)
    slices, _ = slicer.slice(data, None)
    assert len(slices) == 20
