import numpy as np
import pytest
from utils import create_random_input
import tonic.transforms as transforms


def test_time_reversal_spatial_jitter():
    """
    Test the combination of time reversal and spatial jitter transformations.
    """
    orig_events, sensor_size = create_random_input()

    transform = transforms.Compose([
        transforms.RandomTimeReversal(p=1),
        transforms.SpatialJitter(
            sensor_size=sensor_size, var_x=3, var_y=3, sigma_xy=0, clip_outliers=False
        ),
    ])
    events = transform(orig_events)

    assert "RandomTimeReversal" in str(transform)
    assert "SpatialJitter" in str(transform)
    assert len(events) == len(orig_events), "Number of events should be the same."
    assert np.isclose(events["x"], orig_events["x"], atol=3).any(), "Spatial jitter should be within chosen variance x."
    assert (events["x"] != orig_events["x"]).any(), "X coordinates should be different."
    assert np.isclose(events["y"], orig_events["y"], atol=3).any(), "Spatial jitter should be within chosen variance y."
    assert (events["y"] != orig_events["y"]).any(), "Y coordinates should be different."
    assert np.array_equal(orig_events["p"], np.invert(events["p"][::-1].astype(bool)))
    assert np.array_equal(orig_events["t"], np.max(orig_events["t"]) - events["t"][::-1])
    assert events is not orig_events


def test_dropout_flip_ud():
    """
    Test the combination of event dropout and vertical flip transformations.
    """
    orig_events, sensor_size = create_random_input()

    transform = transforms.Compose([
        transforms.DropEvent(p=0.5),
        transforms.RandomFlipUD(sensor_size=sensor_size, p=1),
    ])
    events = transform(orig_events)

    expected_events_count = int((1 - 0.5) * len(orig_events))
    assert len(events) == expected_events_count, "Event dropout should result in expected number of events."
    assert np.allclose(events["t"], np.sort(events["t"])), "Temporal order should be maintained."

    first_dropped_index = np.where(events["t"][0] == orig_events["t"])[0][0]
    assert sensor_size[1] - 1 - orig_events["y"][first_dropped_index] == events["y"][0], (
        "When flipping up and down y must map to the opposite pixel, i.e. y' = sensor height - y - 1"
    )
    assert events is not orig_events


def test_time_skew_flip_polarity_flip_lr():
    """
    Test the combination of time skew, polarity flip, and horizontal flip transformations.
    """
    orig_events, sensor_size = create_random_input()

    transform = transforms.Compose([
        transforms.TimeSkew(coefficient=1.5, offset=0),
        transforms.RandomFlipPolarity(p=1),
        transforms.RandomFlipLR(sensor_size=sensor_size, p=1),
    ])
    events = transform(orig_events)

    assert len(events) == len(orig_events)
    assert (events["t"] >= orig_events["t"]).all()
    assert np.min(events["t"]) >= 0
    assert (events["p"] == np.invert(orig_events["p"].astype(bool)).astype(int)).all(), "Polarities should be flipped."
    assert (sensor_size[0] - 1 - events["x"][0] == orig_events["x"][0]), (
        "When flipping left and right x must map to the opposite pixel, i.e. x' = sensor width - x - 1"
    )
    assert events is not orig_events