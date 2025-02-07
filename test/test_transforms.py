import itertools

import numpy as np
import pytest
import tonic.transforms as transforms
from utils import create_random_input


@pytest.mark.parametrize(
    "sensor_size, size",
    itertools.product(((120, 120, 2), (120, 30, 2)), (50, (50, 50))),
)
def test_center_crop(sensor_size, size):
    orig_events, _ = create_random_input(sensor_size=sensor_size)
    transform = transforms.CenterCrop(sensor_size=sensor_size, size=size)
    events = transform(orig_events)

    if isinstance(size, int):
        size = (size, size)
    assert all(0 <= events["x"]) and all(events["x"] < size[0])
    assert all(0 <= events["y"]) and all(events["y"] < size[1])


@pytest.mark.parametrize("min, max", itertools.product((0, 1000), (None, 5000)))
def test_crop_time(min, max):
    orig_events, sensor_size = create_random_input()
    transform = transforms.CropTime(min=min, max=max)
    events = transform(orig_events)

    assert events is not orig_events
    if min is not None:
        assert not events["t"][0] < min
    if max is not None:
        assert not events["t"][-1] > max


@pytest.mark.parametrize("filter_time", [10000, 5000])
def test_transform_denoise(filter_time):
    orig_events, sensor_size = create_random_input()
    transform = transforms.Denoise(filter_time=filter_time)
    events = transform(orig_events)

    assert len(events) > 0
    assert len(events) < len(orig_events)
    assert np.isin(events, orig_events).all()
    assert events is not orig_events


@pytest.mark.parametrize("p", [0.2, 0.5, (0.1, 0.6)])
def test_transform_drop_events(p):
    orig_events, sensor_size = create_random_input()

    if isinstance(p, tuple):
        sampled_p = transforms.DropEvent.get_params(p=p)
        assert p[0] <= sampled_p <= p[1]
        events = transforms.functional.drop_event_numpy(
            events=orig_events, drop_probability=sampled_p
        )
        p = sampled_p
    else:
        transform = transforms.DropEvent(p=p)
        events = transform(orig_events)

    assert np.isclose(events.shape[0], round((1 - p) * orig_events.shape[0]))
    assert np.isclose(np.sum((events["t"] - np.sort(events["t"])) ** 2), 0)
    assert events is not orig_events


@pytest.mark.parametrize(
    "duration_ratio", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)
def test_transform_drop_events_by_time(duration_ratio):
    orig_events, sensor_size = create_random_input()
    transform = transforms.DropEventByTime(duration_ratio=duration_ratio)
    events = transform(orig_events)

    assert len(events) < len(orig_events)
    t_start = orig_events["t"].min()
    t_end = orig_events["t"].max()
    duration = (t_end - t_start) * duration_ratio
    diffs = np.diff(events["t"])
    assert np.any(diffs >= duration)


@pytest.mark.parametrize("area_ratio", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def test_transform_drop_events_by_area(area_ratio):
    orig_events, sensor_size = create_random_input()
    transform = transforms.DropEventByArea(sensor_size, area_ratio)
    events = transform(orig_events)

    assert len(events) < len(orig_events)
    cut_w = int(area_ratio * sensor_size[0])
    cut_h = int(area_ratio * sensor_size[1])
    to_im = transforms.ToImage(sensor_size)
    frame = to_im(events)
    orig_frame = to_im(orig_events)
    cmp = frame - orig_frame
    dropped_events = len(orig_events) - len(events)

    dropped_area_found = False
    for bbx1 in range(sensor_size[0] - cut_w):
        bbx2 = bbx1 + cut_w
        for bby1 in range(sensor_size[1] - cut_h):
            bby2 = bby1 + cut_h
            if abs(np.sum(cmp[:, bby1:bby2, bbx1:bbx2])) == dropped_events:
                dropped_area_found = True
                break

    assert dropped_area_found


def test_transform_decimation():
    orig_events, sensor_size = create_random_input(sensor_size=(1, 1, 2), n_events=1000)
    transform = transforms.Decimation(n=10)
    events = transform(orig_events)
    assert len(events) == 100


def test_random_drop_pixel():
    orig_events, sensor_size = create_random_input(
        n_events=40000, sensor_size=(15, 15, 2)
    )
    transform = transforms.RandomDropPixel(p=0.5, sensor_size=sensor_size)
    remaining_pixels = [
        len(np.unique(transform(orig_events)[["x", "y"]])) for _ in range(10)
    ]
    n_pixels = np.mean(remaining_pixels)
    assert len(transform(orig_events)) < len(orig_events)
    assert np.isclose(n_pixels, sensor_size[0] * sensor_size[1] * 0.5, atol=10)


def test_random_drop_pixel_raster():
    orig_raster = np.random.randint(0, 10, (50, 2, 100, 200))
    orig_frame = np.random.randint(0, 10, (2, 100, 200))
    transform = transforms.RandomDropPixel(p=0.2)
    raster = transform(orig_raster.copy())
    frame = transform(orig_frame.copy())
    assert np.isclose(raster.sum() / orig_raster.sum(), 0.8, atol=0.05)
    assert np.isclose(frame.sum() / orig_frame.sum(), 0.8, atol=0.05)


@pytest.mark.parametrize(
    "coordinates, hot_pixel_frequency",
    [(((9, 11), (10, 12), (11, 13)), None), (None, 10000)],
)
def test_transform_drop_pixel(coordinates, hot_pixel_frequency):
    orig_events, sensor_size = create_random_input(sensor_size=(20, 20, 2))
    orig_events = np.concatenate((orig_events, np.ones(10000, dtype=orig_events.dtype)))
    orig_events = orig_events[np.argsort(orig_events["t"])]
    transform = transforms.DropPixel(
        coordinates=coordinates, hot_pixel_frequency=hot_pixel_frequency
    )
    events = transform(orig_events)

    assert len(events) < len(orig_events)
    if coordinates:
        for x, y in coordinates:
            assert not np.logical_and(events["x"] == x, events["y"] == y).sum()
    if hot_pixel_frequency:
        assert not np.logical_and(events["x"] == 1, events["y"] == 1).sum()
    assert events is not orig_events


@pytest.mark.parametrize(
    "hot_pixel_frequency, event_max_freq",
    [(59, 60), (10, 60)],
)
def test_transform_drop_pixel_unequal_sensor(hot_pixel_frequency, event_max_freq):
    orig_events, sensor_size = create_random_input(
        n_events=40000, sensor_size=(15, 20, 2)
    )
    orig_events = orig_events.tolist()
    orig_events += [
        (0, 0, int(t * 1e3), 1) for t in np.arange(1, 1e6, 1e3 / event_max_freq)
    ]
    orig_events += [
        (0, 19, int(t * 1e3), 1) for t in np.arange(1, 1e6, 1e3 / event_max_freq)
    ]
    orig_events += [
        (14, 0, int(t * 1e3), 1) for t in np.arange(1, 1e6, 1e3 / event_max_freq)
    ]
    orig_events += [
        (14, 19, int(t * 1e3), 1) for t in np.arange(1, 1e6, 1e3 / event_max_freq)
    ]
    orig_events = np.asarray(
        orig_events, np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    )
    transform = transforms.DropPixel(
        coordinates=None, hot_pixel_frequency=hot_pixel_frequency
    )
    events = transform(orig_events)
    assert len(np.where((events["x"] == 0) & (events["y"] == 0))[0]) == 0
    assert len(np.where((events["x"] == 14) & (events["y"] == 0))[0]) == 0
    assert len(np.where((events["x"] == 0) & (events["y"] == 19))[0]) == 0
    assert len(np.where((events["x"] == 14) & (events["y"] == 19))[0]) == 0


@pytest.mark.parametrize(
    "coordinates, hot_pixel_frequency",
    [(((9, 11), (10, 12), (11, 13)), None), (None, 10000)],
)
def test_transform_drop_pixel_empty(coordinates, hot_pixel_frequency):
    orig_events, sensor_size = create_random_input(n_events=0, sensor_size=(15, 20, 2))
    transform = transforms.DropPixel(
        coordinates=None, hot_pixel_frequency=hot_pixel_frequency
    )
    events = transform(orig_events)
    assert len(events) == len(orig_events)
    transform = transforms.DropPixel(coordinates=coordinates, hot_pixel_frequency=None)
    events = transform(orig_events)
    assert len(events) == len(orig_events)


@pytest.mark.parametrize(
    "coordinates, hot_pixel_frequency",
    [(((199, 11), (199, 12), (11, 13)), None), (None, 5000)],
)
def test_transform_drop_pixel_raster(coordinates, hot_pixel_frequency):
    raster_test = np.random.randint(0, 100, (50, 2, 100, 200))
    frame_test = np.random.randint(0, 100, (2, 100, 200))
    transform = transforms.DropPixel(
        coordinates=coordinates, hot_pixel_frequency=hot_pixel_frequency
    )
    raster = transform(raster_test)
    frame = transform(frame_test)

    if coordinates:
        for x, y in coordinates:
            assert raster[:, :, y, x].sum() == 0
            assert frame[:, y, x].sum() == 0
    if hot_pixel_frequency:
        merged_polarity_raster = raster.sum(0).sum(0)
        merged_polarity_frame = frame.sum(0)
        assert not merged_polarity_frame[merged_polarity_frame > 5000].sum().sum()
        assert not merged_polarity_raster[merged_polarity_raster > 5000].sum().sum()


@pytest.mark.parametrize(
    "time_factor, spatial_factor, target_size",
    [(1, 0.25, None), (1e-3, (1, 2), None), (1, 1, (5, 5))],
)
def test_transform_downsample(time_factor, spatial_factor, target_size):
    orig_events, sensor_size = create_random_input()
    transform = transforms.Downsample(
        sensor_size=sensor_size,
        time_factor=time_factor,
        spatial_factor=spatial_factor,
        target_size=target_size,
    )
    events = transform(orig_events)

    if not isinstance(spatial_factor, tuple):
        spatial_factor = (spatial_factor, spatial_factor)

    if target_size is None:
        assert np.array_equal(
            (orig_events["t"] * time_factor).astype(orig_events["t"].dtype), events["t"]
        )
        assert np.array_equal(
            np.floor(orig_events["x"] * spatial_factor[0]), events["x"]
        )
        assert np.array_equal(
            np.floor(orig_events["y"] * spatial_factor[1]), events["y"]
        )

    else:
        spatial_factor_test = np.asarray(target_size) / sensor_size[:-1]
        assert np.array_equal(
            np.floor(orig_events["x"] * spatial_factor_test[0]), events["x"]
        )
        assert np.array_equal(
            np.floor(orig_events["y"] * spatial_factor_test[1]), events["y"]
        )

    assert events is not orig_events


@pytest.mark.parametrize(
    "target_size, dt, downsampling_method, noise_threshold, differentiator_time_bins",
    [((50, 50), 0.05, "integrator", 1, None), ((20, 15), 5, "differentiator", 3, 1)],
)
def test_transform_event_downsampling(
    target_size, dt, downsampling_method, noise_threshold, differentiator_time_bins
):
    orig_events, sensor_size = create_random_input()
    transform = transforms.EventDownsampling(
        sensor_size=sensor_size,
        target_size=target_size,
        dt=dt,
        downsampling_method=downsampling_method,
        noise_threshold=noise_threshold,
        differentiator_time_bins=differentiator_time_bins,
    )
    events = transform(orig_events)

    assert len(events) <= len(orig_events)
    assert np.logical_and(
        np.all(events["x"] <= target_size[0]), np.all(events["y"] <= target_size[1])
    )
    assert events is not orig_events


@pytest.mark.parametrize("target_size", [(50, 50), (10, 5)])
def test_transform_random_crop(target_size):
    orig_events, sensor_size = create_random_input()
    transform = transforms.RandomCrop(sensor_size=sensor_size, target_size=target_size)
    events = transform(orig_events)

    assert np.all(events["x"]) < target_size[0] and np.all(
        events["y"] < target_size[1]
    ), "Cropping needs to map the events into the new space."
    assert events is not orig_events


@pytest.mark.parametrize("p", [1.0, 0])
def test_transform_flip_lr(p):
    orig_events, sensor_size = create_random_input()
    transform = transforms.RandomFlipLR(sensor_size=sensor_size, p=p)
    events = transform(orig_events)

    if p == 1:
        assert ((sensor_size[0] - 1) - orig_events["x"] == events["x"]).all()
    else:
        assert np.array_equal(orig_events, events)
    assert events is not orig_events


@pytest.mark.parametrize("p", [1.0, 0])
def test_transform_flip_polarity(p):
    orig_events, sensor_size = create_random_input()
    transform = transforms.RandomFlipPolarity(p=p)
    events = transform(orig_events)

    if p == 1:
        assert np.array_equal(np.invert(orig_events["p"].astype(bool)), events["p"])
    else:
        assert np.array_equal(orig_events["p"], events["p"])
    assert events is not orig_events


@pytest.mark.parametrize("p", [1.0, 0])
def test_transform_flip_polarity_bools(p):
    orig_events, sensor_size = create_random_input(
        dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", bool)])
    )
    transform = transforms.RandomFlipPolarity(p=p)
    events = transform(orig_events)

    if p == 1:
        assert np.array_equal(np.invert(orig_events["p"].astype(bool)), events["p"])
    else:
        assert np.array_equal(orig_events["p"], events["p"])
    assert events is not orig_events


@pytest.mark.parametrize("p", [1.0, 0])
def test_transform_flip_ud(p):
    orig_events, sensor_size = create_random_input()
    transform = transforms.RandomFlipUD(sensor_size=sensor_size, p=p)
    events = transform(orig_events)

    if p == 1:
        assert np.array_equal((sensor_size[1] - 1) - orig_events["y"], events["y"])
    else:
        assert np.array_equal(orig_events, events)
    assert events is not orig_events


def test_transform_merge_polarities():
    orig_events, sensor_size = create_random_input()
    transform = transforms.MergePolarities()
    events = transform(orig_events)
    assert len(np.unique(orig_events["p"])) == 2
    assert len(np.unique(events["p"])) == 1
    assert events is not orig_events


def test_transform_numpy_array():
    orig_events, sensor_size = create_random_input()
    transform = transforms.NumpyAsType(int)
    events = transform(orig_events)
    assert events.dtype == int
    assert events is not orig_events


def test_transform_numpy_array_unstructured():
    orig_events, sensor_size = create_random_input()
    transform = transforms.NumpyAsType(int)
    events = transform(orig_events)
    assert events.dtype == int
    assert events is not orig_events


@pytest.mark.parametrize("delta", [10000, 5000, (2000, 5000)])
def test_transform_refractory_period(delta):
    orig_events, sensor_size = create_random_input()

    if isinstance(delta, tuple):
        sampled_delta = transforms.RefractoryPeriod.get_params(delta)
        assert delta[0] <= sampled_delta <= delta[1]
        assert float(sampled_delta).is_integer()
        events = transforms.functional.refractory_period_numpy(
            events=orig_events, refractory_period=sampled_delta
        )

    else:
        transform = transforms.RefractoryPeriod(delta=delta)
        events = transform(orig_events)

    assert len(events) > 0
    assert len(events) < len(orig_events)
    assert np.isin(events, orig_events).all()
    assert events.dtype == events.dtype
    assert events is not orig_events


@pytest.mark.parametrize(
    "variance, clip_outliers", [(30, False), (100, True), (3.5, True), (0.8, False)]
)
def test_transform_spatial_jitter(variance, clip_outliers):
    orig_events, sensor_size = create_random_input()

    transform = transforms.SpatialJitter(
        sensor_size=sensor_size,
        var_x=variance,
        var_y=variance,
        sigma_xy=0,
        clip_outliers=clip_outliers,
    )

    events = transform(orig_events)

    if not clip_outliers:
        assert len(events) == len(orig_events)
        assert (events["t"] == orig_events["t"]).all()
        assert (events["p"] == orig_events["p"]).all()
        assert (events["x"] != orig_events["x"]).any()
        assert (events["y"] != orig_events["y"]).any()
        assert np.isclose(events["x"].all(), orig_events["x"].all(), atol=2 * variance)
        assert np.isclose(events["y"].all(), orig_events["y"].all(), atol=2 * variance)

        assert (
            events["x"] - orig_events["x"]
            == (events["x"] - orig_events["x"]).astype(int)
        ).all()

        assert (
            events["y"] - orig_events["y"]
            == (events["y"] - orig_events["y"]).astype(int)
        ).all()

    else:
        assert len(events)


@pytest.mark.parametrize(
    "std, clip_negative, sort_timestamps",
    [(10, True, True), (50, False, False), (0, True, False)],
)
def test_transform_time_jitter(std, clip_negative, sort_timestamps):
    orig_events, sensor_size = create_random_input()

    transform = transforms.TimeJitter(
        std=std, clip_negative=clip_negative, sort_timestamps=sort_timestamps
    )

    events = transform(orig_events)

    if clip_negative:
        assert (events["t"] >= 0).all()
    else:
        assert len(events) == len(orig_events)
    if sort_timestamps:
        np.testing.assert_array_equal(events["t"], np.sort(events["t"]))
    if not sort_timestamps and not clip_negative:
        np.testing.assert_array_equal(events["x"], orig_events["x"])
        np.testing.assert_array_equal(events["y"], orig_events["y"])
        np.testing.assert_array_equal(events["p"], orig_events["p"])
        assert (
            events["t"] - orig_events["t"]
            == (events["t"] - orig_events["t"]).astype(int)
        ).all()
    assert events is not orig_events


@pytest.mark.parametrize("p", [1, 0])
def test_transform_time_reversal(p):
    orig_events, sensor_size = create_random_input()

    original_t = orig_events["t"][0]
    max_t = np.max(orig_events["t"])

    transform = transforms.RandomTimeReversal(p=p)
    events = transform(orig_events)

    if p == 1:
        assert np.array_equal(orig_events["t"], max_t - events["t"][::-1])
        assert np.array_equal(
            orig_events["p"],
            np.invert(events["p"][::-1].astype(bool)),
        )
    elif p == 0:
        assert np.array_equal(orig_events, events)
    assert events is not orig_events


@pytest.mark.parametrize("p", [1, 0])
def test_random_reversal_raster(p):
    orig_events, sensor_size = create_random_input()
    to_raster = transforms.ToFrame(sensor_size=sensor_size, n_event_bins=100)
    # raster in shape [t, p, h, w]
    orig_raster = to_raster(orig_events)

    transform = transforms.RandomTimeReversal(p=p)
    raster = transform(orig_raster)

    if p == 1:
        assert np.array_equal(raster, orig_raster[::-1, ::-1, ...])
    elif p == 0:
        assert np.array_equal(raster, orig_raster)
    assert raster is not orig_raster


@pytest.mark.parametrize("coefficient, offset", [(3.1, 100), (0.3, 0), (2.7, 10)])
def test_transform_time_skew(coefficient, offset):
    orig_events, sensor_size = create_random_input()

    transform = transforms.TimeSkew(coefficient=coefficient, offset=offset)

    events = transform(orig_events)

    assert len(events) == len(orig_events)
    assert np.min(events["t"]) >= offset
    assert (events["t"] == (events["t"]).astype(int)).all()
    assert all((orig_events["t"] * coefficient + offset).astype(int) == events["t"])
    assert events is not orig_events


@pytest.mark.parametrize("n", [100, 0, (10, 100)])
def test_transform_uniform_noise(n):
    orig_events, sensor_size = create_random_input()

    if type(n) == tuple:
        sampled_n = transforms.UniformNoise.get_params(n=n)
        assert n[0] <= sampled_n <= n[1]
        assert float(sampled_n).is_integer()
        events = transforms.functional.uniform_noise_numpy(
            events=orig_events, sensor_size=sensor_size, n=sampled_n
        )
        assert len(events) == len(orig_events) + sampled_n

    else:
        transform = transforms.UniformNoise(sensor_size=sensor_size, n=n)
        events = transform(orig_events)
        assert len(events) == len(orig_events) + n

    assert np.isin(orig_events, events).all()
    assert np.isclose(
        np.sum((events["t"] - np.sort(events["t"])) ** 2), 0
    ), "Event noise should maintain temporal order."
    assert events is not orig_events


@pytest.mark.parametrize("n", [100, 0, (10, 100)])
def test_transform_uniform_noise_empty(n):
    orig_events, sensor_size = create_random_input(n_events=0)
    assert len(orig_events) == 0

    transform = transforms.UniformNoise(sensor_size=sensor_size, n=n)
    events = transform(orig_events)
    assert len(events) == 0  # check returns an empty array, independent of n.


def test_transform_time_alignment():
    orig_events, sensor_size = create_random_input()

    transform = transforms.TimeAlignment()

    events = transform(orig_events)

    assert np.min(events["t"]) == 0
    assert events is not orig_events


def test_toframe_empty():
    orig_events, sensor_size = create_random_input(n_events=0)
    assert len(orig_events) == 0

    with pytest.raises(
        ValueError
    ):  # check that empty array raises error if no slicing method is specified
        transform = transforms.ToFrame(sensor_size=sensor_size)
        frame = transform(orig_events)

    n_event_bins = 100
    transform = transforms.ToFrame(sensor_size=sensor_size, n_event_bins=n_event_bins)
    frame = transform(orig_events)
    assert frame.shape == (n_event_bins, sensor_size[2], sensor_size[0], sensor_size[1])
    assert frame.sum() == 0

    n_time_bins = 100
    transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins)
    frame = transform(orig_events)
    assert frame.shape == (n_time_bins, sensor_size[2], sensor_size[0], sensor_size[1])
    assert frame.sum() == 0

    event_count = 1e3
    transform = transforms.ToFrame(sensor_size=sensor_size, event_count=event_count)
    frame = transform(orig_events)
    assert frame.shape == (1, sensor_size[2], sensor_size[0], sensor_size[1])
    assert frame.sum() == 0

    time_window = 1e3
    transform = transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)
    frame = transform(orig_events)
    assert frame.shape == (1, sensor_size[2], sensor_size[0], sensor_size[1])
    assert frame.sum() == 0
