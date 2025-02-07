from typing import Any, Tuple

import numpy as np


def create_random_input(
    sensor_size: Tuple[int, int, int] = (200, 100, 2),
    n_events: int = 10000,
    dtype: Any = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]),
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Generates a set of random events for sensor simulation.

    Parameters:
    - sensor_size: A tuple representing the sensor dimensions (width, height, polarity).
    - n_events: The number of random events to generate.
    - dtype: The data type for the numpy structured array of events.

    Returns:
    - A tuple containing:
        - events: A numpy structured array with random event data.
        - sensor_size: The dimensions of the sensor.
    """
    assert all(
        key in dtype.names for key in ["x", "t", "p"]
    ), "dtype must include 'x', 't', and 'p' fields"

    events = np.zeros(n_events, dtype=dtype)
    events["x"] = np.random.randint(0, sensor_size[0], n_events)
    events["p"] = np.random.randint(0, sensor_size[2], n_events)
    events["t"] = np.sort(np.random.randint(0, 1_000_000, n_events))

    if "y" in dtype.names:
        events["y"] = np.random.randint(0, sensor_size[1], n_events)

    return events, sensor_size
