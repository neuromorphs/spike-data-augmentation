import pytest
import numpy as np
import tonic.transforms as transforms
import utils


class TestTransforms:
    @pytest.mark.parametrize(
        "ordering, target_size", [("xytp", (50, 50)), ("typx", (10, 5))]
    )
    def test_transform_crop(self, ordering, target_size):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.Crop(target_size=target_size)

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        assert np.all(events[:, x_index]) < target_size[0] and np.all(
            events[:, y_index] < target_size[1]
        ), "Cropping needs to map the events into the new space"
        assert (
            images.shape[2] == target_size[0] and images.shape[1] == target_size[1]
        ), "Cropping needs to map the images into the new space"

    @pytest.mark.parametrize(
        "ordering, drop_probability, random_drop_probability",
        [("xytp", 0.2, False), ("typx", 0.5, True)],
    )
    def test_transform_drop_events(
        self, ordering, drop_probability, random_drop_probability
    ):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.DropEvent(
            drop_probability=drop_probability,
            random_drop_probability=random_drop_probability,
        )

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        if random_drop_probability:
            assert events.shape[0] >= (1 - drop_probability) * orig_events.shape[0], (
                "Event dropout with random drop probability should result in less than "
                " drop_probability*len(original) events dropped out."
            )
        else:
            assert np.isclose(
                events.shape[0], (1 - drop_probability) * orig_events.shape[0]
            ), (
                "Event dropout should result in drop_probability*len(original) events"
                " dropped out."
            )
        assert np.isclose(
            np.sum((events[:, t_index] - np.sort(events[:, t_index])) ** 2), 0
        ), "Event dropout should maintain temporal order."

    @pytest.mark.parametrize(
        "ordering, time_factor, spatial_factor", [("xytp", 1, 0.25), ("typx", 1e-3, 1)]
    )
    def test_transform_downsample(self, ordering, time_factor, spatial_factor):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.Downsample(
            time_factor=time_factor, spatial_factor=spatial_factor
        )

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        assert np.array_equal(orig_events[:, t_index] * time_factor, events[:, t_index])
        assert np.array_equal(
            np.floor(orig_events[:, x_index] * spatial_factor), events[:, x_index]
        )
        assert np.array_equal(
            np.floor(orig_events[:, y_index] * spatial_factor), events[:, y_index]
        )

    @pytest.mark.parametrize(
        "ordering, flip_probability", [("xytp", 1.0), ("typx", 1.0)]
    )
    def test_transform_flip_lr(self, ordering, flip_probability):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.FlipLR(flip_probability=flip_probability)

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        assert (
            (sensor_size[0] - 1) - orig_events[:, x_index] == events[:, x_index]
        ).all(), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )

    @pytest.mark.parametrize("ordering, flip_probability", [("xytp", 1.0), ("typx", 0)])
    def test_transform_flip_polarity(self, ordering, flip_probability):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)
        transform = transforms.FlipPolarity(flip_probability=flip_probability)

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        if flip_probability == 1:
            assert np.array_equal(orig_events[:, p_index] * -1, events[:, p_index]), (
                "When flipping polarity with probability 1, all event polarities must"
                " flip"
            )
        else:
            assert np.array_equal(orig_events[:, p_index], events[:, p_index]), (
                "When flipping polarity with probability 0, no event polarities must"
                " flip"
            )

    @pytest.mark.parametrize(
        "ordering, flip_probability", [("xytp", 1.0), ("typx", 1.0)]
    )
    def test_transform_flip_ud(self, ordering, flip_probability):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)
        transform = transforms.FlipUD(flip_probability=flip_probability)

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        assert np.array_equal(
            (sensor_size[1] - 1) - orig_events[:, y_index], events[:, y_index]
        ), (
            "When flipping left and right x must map to the opposite pixel, i.e. x' ="
            " sensor width - x"
        )

    @pytest.mark.parametrize("ordering, filter_time", [("xytp", 1000), ("typx", 500)])
    def test_transform_denoise(self, ordering, filter_time):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        transform = transforms.Denoise(filter_time=filter_time)

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        assert len(events) > 0, "Not all events should be filtered"
        assert len(events) < len(
            orig_events
        ), "Result should be fewer events than original event stream"
        assert np.isin(events, orig_events).all(), (
            "Denoising should not add additional events that were not present in"
            " original event stream"
        )

    @pytest.mark.parametrize(
        "ordering, refractory_period", [("xytp", 1000), ("typx", 50)]
    )
    def test_transform_refractory_period(self, ordering, refractory_period):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        transform = transforms.RefractoryPeriod(refractory_period=refractory_period)

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        assert len(events) > 0, "Not all events should be filtered"
        assert len(events) < len(
            orig_events
        ), "Result should be fewer events than original event stream"
        assert np.isin(
            events, orig_events
        ).all(), (
            "Added additional events that were not present in original event stream"
        )
        assert events.dtype == events.dtype

    @pytest.mark.parametrize(
        "ordering, variance, integer_jitter, clip_outliers",
        [
            ("xytp", 30, True, False),
            ("typx", 100, True, True),
            ("typx", 3.5, False, True),
            ("typx", 0.8, False, False),
        ],
    )
    def test_transform_spatial_jitter(
        self, ordering, variance, integer_jitter, clip_outliers
    ):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)
        transform = transforms.SpatialJitter(
            variance_x=variance,
            variance_y=variance,
            sigma_x_y=0,
            integer_jitter=integer_jitter,
            clip_outliers=clip_outliers,
        )

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        if not clip_outliers:
            assert len(events) == len(orig_events)
            assert (events[:, t_index] == orig_events[:, t_index]).all()
            assert (events[:, p_index] == orig_events[:, p_index]).all()
            assert (events[:, x_index] != orig_events[:, x_index]).any()
            assert (events[:, y_index] != orig_events[:, y_index]).any()
            assert np.isclose(
                events[:, x_index].all(),
                orig_events[:, x_index].all(),
                atol=2 * variance,
            )
            assert np.isclose(
                events[:, y_index].all(),
                orig_events[:, y_index].all(),
                atol=2 * variance,
            )

            if integer_jitter:
                assert (
                    events[:, x_index] - orig_events[:, x_index]
                    == (events[:, x_index] - orig_events[:, x_index]).astype(int)
                ).all()

                assert (
                    events[:, y_index] - orig_events[:, y_index]
                    == (events[:, y_index] - orig_events[:, y_index]).astype(int)
                ).all()

        else:
            assert len(events) < len(orig_events)

    @pytest.mark.parametrize(
        "ordering, std, integer_jitter, clip_negative, sort_timestamps",
        [
            ("xytp", 10, False, True, True),
            ("typx", 50, True, False, False),
            ("pxty", 0, True, True, False),
        ],
    )
    def test_transform_time_jitter(
        self, ordering, std, integer_jitter, clip_negative, sort_timestamps
    ):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        # we do this to ensure integer timestamps before testing for int jittering
        if integer_jitter:
            orig_events[:, t_index] = orig_events[:, t_index].round()

        transform = transforms.TimeJitter(
            std=std,
            integer_jitter=integer_jitter,
            clip_negative=clip_negative,
            sort_timestamps=sort_timestamps,
        )

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        if clip_negative:
            assert (events[:, t_index] >= 0).all()
        else:
            assert len(events) == len(orig_events)
        if sort_timestamps:
            np.testing.assert_array_equal(
                events[:, t_index], np.sort(events[:, t_index])
            )
        if not sort_timestamps and not clip_negative:
            np.testing.assert_array_equal(events[:, x_index], orig_events[:, x_index])
            np.testing.assert_array_equal(events[:, y_index], orig_events[:, y_index])
            np.testing.assert_array_equal(events[:, p_index], orig_events[:, p_index])
            if integer_jitter:
                assert (
                    events[:, t_index] - orig_events[:, t_index]
                    == (events[:, t_index] - orig_events[:, t_index]).astype(int)
                ).all()

    @pytest.mark.parametrize(
        "ordering, flip_probability", [("xytp", 1000), ("typx", 50)]
    )
    def test_transform_time_reversal(self, ordering, flip_probability):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        original_t = orig_events[0, t_index]
        original_p = orig_events[0, p_index]

        max_t = np.max(orig_events[:, t_index])

        transform = transforms.TimeReversal(flip_probability=flip_probability,)

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        same_time = np.isclose(max_t - original_t, events[0, t_index])
        same_polarity = np.isclose(events[0, p_index], -1.0 * original_p)

        assert same_time, "When flipping time must map t_i' = max(t) - t_i"
        assert same_polarity, "When flipping time polarity should be flipped"
        assert events.dtype == events.dtype

    @pytest.mark.parametrize(
        "ordering, offset, coefficient, integer_time",
        [("xytp", 100, 3.1, True), ("typx", 0, 0.7, False), ("ptyx", 10, 2.7, False)],
    )
    def test_transform_time_skew(self, ordering, offset, coefficient, integer_time):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.TimeSkew(
            coefficient=coefficient, offset=offset, integer_time=integer_time,
        )

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        assert len(events) == len(orig_events)
        assert np.min(events[:, t_index]) >= offset
        if integer_time:
            assert (events[:, t_index] == (events[:, t_index]).astype(int)).all()

        else:
            if coefficient > 1:
                assert (events[:, t_index] - offset > orig_events[:, t_index]).all()

            if coefficient < 1:
                assert (events[:, t_index] - offset < orig_events[:, t_index]).all()

            assert (events[:, t_index] != (events[:, t_index]).astype(int)).any()

    @pytest.mark.parametrize("ordering", ["xytp", "typx"])
    def test_transform_uniform_noise(self, ordering):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)

        transform = transforms.UniformNoise(
            scaling_factor_to_micro_sec=1000000, noise_density=1e-8,
        )

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            images=orig_images.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            multi_image=is_multi_image,
        )

        assert len(events) > len(orig_events)
        assert np.isin(orig_events, events).all()

    @pytest.mark.parametrize("ordering",[("xytp"), ("typx")])
    def test_transform_time_alignment(self, ordering):
        (
            orig_events,
            orig_images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        x_index, y_index, t_index, p_index = utils.findXytpPermutation(ordering)

        transform = transforms.TimeAlignment()

        events, images, sensor_size = transform(
            events=orig_events.copy(),
            sensor_size=sensor_size,
            ordering=ordering,
            images=orig_images.copy(),
            multi_image=is_multi_image,
        )

        assert np.min(events[:, t_index]) == 0
