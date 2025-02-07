import numpy as np
import pytest
import tonic.transforms as transforms
from utils import create_random_input


@pytest.mark.parametrize(
    "time_window, event_count, n_time_bins, n_event_bins, overlap, include_incomplete, sensor_size",
    [
        (20000, None, None, None, 0, False, (40, 20, 2)),
        (20000, None, None, None, 200, True, (40, 20, 1)),
        (10000, None, None, None, 100, True, (40, 20, 3)),
        (None, 2000, None, None, 0, False, (40, 20, 2)),
        (None, 2000, None, None, 200, True, (20, 20, 1)),
        (None, 2000, None, None, 100, True, (10, 20, 2)),
        (None, None, 5, None, 0, False, (40, 20, 2)),
        (None, None, 5, None, 0.1, False, (10, 20, 2)),
        (None, None, 5, None, 0.25, True, (40, 20, 2)),
        (None, None, None, 5, 0, True, (40, 20, 2)),
        (None, None, None, 5, 0.1, False, (40, 20, 2)),
        (None, None, None, 5, 0.25, False, (10, 20, 1)),
    ],
)
def test_representation_frame(
    time_window: int,
    event_count: int,
    n_time_bins: int,
    n_event_bins: int,
    overlap: float,
    include_incomplete: bool,
    sensor_size: tuple,
):
    """Test the frame representation transformation with various parameters."""
    n_events = 10000
    orig_events, _ = create_random_input(sensor_size=sensor_size, n_events=n_events)

    transform = transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=time_window,
        event_count=event_count,
        n_time_bins=n_time_bins,
        n_event_bins=n_event_bins,
        overlap=overlap,
        include_incomplete=include_incomplete,
    )

    frames = transform(orig_events)

    assert frames.shape[1:] == sensor_size[::-1]
    times = orig_events["t"]
    if time_window is not None:
        stride = time_window - overlap
        expected_frames = (
            np.ceil(((times[-1] - times[0]) - time_window) / stride) + 1
            if include_incomplete
            else np.floor(((times[-1] - times[0]) - time_window) / stride) + 1
        )
        assert frames.shape[0] == int(expected_frames)

    if event_count is not None:
        assert event_count == frames[0].sum()
        stride = event_count - overlap
        expected_frames = (
            np.ceil((n_events - event_count) / stride) + 1
            if include_incomplete
            else np.floor((n_events - event_count) / stride) + 1
        )
        assert frames.shape[0] == int(expected_frames)

    if n_time_bins is not None:
        assert frames.shape[0] == n_time_bins

    if n_event_bins is not None:
        assert frames.shape[0] == n_event_bins
        assert frames[0].sum() == (1 + overlap) * (n_events // n_event_bins)

    assert frames is not orig_events


@pytest.mark.parametrize(
    "time_window, event_count, n_time_bins, n_event_bins, overlap,"
    " include_incomplete, sensor_size",
    [
        (2000, None, None, None, 0, False, (40, 20, 2)),
        (2000, None, None, None, 200, True, (40, 20, 1)),
        (1000, None, None, None, 100, True, (40, 20, 3)),
        (None, 2000, None, None, 0, False, (40, 20, 2)),
        (None, 2000, None, None, 200, True, (20, 20, 1)),
        (None, 2000, None, None, 100, True, (10, 20, 2)),
        (None, None, 5, None, 0, False, (40, 20, 2)),
        (None, None, 5, None, 0.1, False, (10, 20, 2)),
        (None, None, 5, None, 0.25, True, (40, 20, 2)),
        (None, None, None, 5, 0, True, (40, 20, 2)),
        (None, None, None, 5, 0.1, False, (40, 20, 2)),
        (None, None, None, 5, 0.25, False, (10, 20, 1)),
    ],
)
def test_representation_sparse_tensor(
    time_window,
    event_count,
    n_time_bins,
    n_event_bins,
    overlap,
    include_incomplete,
    sensor_size,
):
    n_events = 10000
    orig_events, sensor_size = create_random_input(
        sensor_size=sensor_size, n_events=n_events
    )

    transform = transforms.ToSparseTensor(
        sensor_size=sensor_size,
        time_window=time_window,
        event_count=event_count,
        n_time_bins=n_time_bins,
        n_event_bins=n_event_bins,
        overlap=overlap,
        include_incomplete=include_incomplete,
    )

    sparse_tensor = transform(orig_events)

    assert sparse_tensor.is_sparse
    assert sparse_tensor.shape[1:] == sensor_size[::-1]

    if time_window is not None:
        stride = time_window - overlap
        times = orig_events["t"]
        if include_incomplete:
            assert sparse_tensor.shape[0] == int(
                np.ceil(((times[-1] - times[0]) - time_window) / stride) + 1
            )
        else:
            assert sparse_tensor.shape[0] == int(
                np.floor(((times[-1] - times[0]) - time_window) / stride) + 1
            )

    if event_count is not None:
        assert event_count == sparse_tensor[0].to_dense().sum()
        stride = event_count - overlap
        if include_incomplete:
            assert sparse_tensor.shape[0] == int(
                np.ceil((n_events - event_count) / stride) + 1
            )
        else:
            assert sparse_tensor.shape[0] == int(
                np.floor((n_events - event_count) / stride) + 1
            )

    if n_time_bins is not None:
        assert sparse_tensor.shape[0] == n_time_bins

    if n_event_bins is not None:
        assert sparse_tensor.shape[0] == n_event_bins
        assert sparse_tensor[0].to_dense().sum() == (1 + overlap) * (
            n_events // n_event_bins
        )

    assert sparse_tensor is not orig_events


def test_representation_inferred_frame():
    """Test frame representation with inferred sensor size."""
    sensor_size = (20, 10, 2)
    orig_events, _ = create_random_input(n_events=30000, sensor_size=sensor_size)
    transform = transforms.ToFrame(sensor_size=None, time_window=25000)
    frames = transform(orig_events)
    assert frames.shape[1:] == sensor_size[::-1]


def test_representation_frame_wrong_sensor_size():
    """Test frame representation with a mismatched sensor size, expecting an error."""
    sensor_size = (20, 10, 2)
    orig_events, _ = create_random_input(n_events=30000, sensor_size=sensor_size)
    transform = transforms.ToFrame(sensor_size=(20, 10, 1), time_window=25000)
    with pytest.raises(ValueError):
        frames = transform(orig_events)


def test_representation_audio_frame():
    """Test frame representation specifically for audio data."""
    sensor_size = (200, 1, 2)
    orig_events, _ = create_random_input(
        sensor_size=sensor_size,
        dtype=np.dtype([("x", int), ("t", int), ("p", int)]),
    )
    transform = transforms.ToFrame(sensor_size=sensor_size, time_window=25000)
    frames = transform(orig_events)
    assert frames.shape[1:] == (sensor_size[2], sensor_size[0])


def test_representation_image():
    """Test image representation from event data."""
    sensor_size = (100, 100, 2)
    orig_events, _ = create_random_input(n_events=10000, sensor_size=sensor_size)
    transform = transforms.ToImage(sensor_size=sensor_size)
    image = transform(orig_events)
    assert image.shape == sensor_size[::-1]


@pytest.mark.parametrize(
    "sensor_size, dt, tau",
    [
        ((40, 15, 2), 10000, 10000),
        ((10, 20, 2), 20000, 1000),
        ((30, 30, 2), 30000, 3000),
    ],
)
def test_representation_time_surface(sensor_size, dt, tau):
    """Test time surface representation with varying parameters."""
    orig_events, sensor_size = create_random_input(
        sensor_size=sensor_size, n_events=10000
    )

    transform = transforms.ToTimesurface(sensor_size=sensor_size, dt=dt, tau=tau)

    surfaces = transform(orig_events)

    duration = orig_events["t"][-1] - orig_events["t"][0]
    assert surfaces.shape[0] == duration // dt

    assert surfaces.shape[1] == 2
    assert surfaces.shape[2] == sensor_size[1]
    assert surfaces.shape[3] == sensor_size[0]
    assert surfaces is not orig_events


@pytest.mark.parametrize(
    "surface_size, cell_size, tau, decay",
    [(7, 9, 100, "lin"), (3, 4, 1000, "exp")],
)
def test_representation_avg_time_surface(surface_size, cell_size, tau, decay):
    """Test averaged time surface representation with various decay functions."""
    orig_events, sensor_size = create_random_input(n_events=1000)

    transform = transforms.ToAveragedTimesurface(
        sensor_size=sensor_size,
        surface_size=surface_size,
        cell_size=cell_size,
        tau=tau,
        decay=decay,
    )

    surfaces = transform(orig_events)
    import math

    assert len(surfaces.shape) == 4
    assert surfaces.shape[0] == math.ceil(sensor_size[0] / cell_size) * math.ceil(
        sensor_size[1] / cell_size
    )
    assert surfaces.shape[1] == sensor_size[2]
    assert surfaces.shape[2] == surface_size
    assert surfaces.shape[3] == surface_size
    assert surfaces is not orig_events


@pytest.mark.parametrize("n_time_bins", [10, 1])
def test_representation_voxel_grid(n_time_bins):
    """Test voxel grid representation with varying number of time bins."""
    orig_events, sensor_size = create_random_input()

    transform = transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=n_time_bins)

    volume = transform(orig_events)

    assert volume.shape == (n_time_bins, 1, *sensor_size[1::-1])
    assert volume is not orig_events


@pytest.mark.parametrize(
    "n_frames, n_bits",
    [(1, 8), (2, 8), (3, 8), (1, 16), (2, 16)],
)
def test_bina_rep(n_frames, n_bits):
    """Test binary representation transformation with varying frames and bit depths."""
    n_events = 10000
    sensor_size = (128, 128, 2)

    orig_events, _ = create_random_input(sensor_size=sensor_size, n_events=n_events)

    transform = transforms.Compose(
        [
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_frames * n_bits),
            transforms.ToBinaRep(
                n_frames=n_frames,
                n_bits=n_bits,
            ),
        ]
    )

    frames = transform(orig_events)

    assert len(frames.shape) == 4
    assert frames.shape[0] == n_frames
    assert frames.shape[1:] == sensor_size[::-1]
    assert frames is not orig_events
