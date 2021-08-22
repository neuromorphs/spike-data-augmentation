from . import functional


class Compose:
    """Composes several transforms together.

    Parameters:
        transforms (list of ``Transform`` objects): list of transform(s) to compose. Even when using a single transform, the Compose wrapper is necessary.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Denoise(),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        for t in self.transforms:
            events, images, sensor_size = t(
                events=events,
                images=images,
                sensor_size=sensor_size,
                ordering=ordering,
                multi_image=multi_image,
            )
        if multi_image and images.all() != None:
            return events, images, sensor_size
        else:
            return events

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Crop:
    """Crops the sensor size to a smaller sensor.
    Removes events outsize of the target sensor and maps.

    x' = x - new_sensor_start_x

    y' = y - new_sensor_start_y

    Parameters:
        target_size: size of the sensor that was used [W',H']
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.crop_numpy(
            events=events,
            sensor_size=sensor_size,
            ordering=ordering,
            images=images,
            multi_image=multi_image,
            target_size=self.target_size,
        )


class Denoise:
    """Drops events that are 'not sufficiently connected to other events in the recording.'
    In practise that means that an event is dropped if no other event occured within a spatial neighbourhood
    of 1 pixel and a temporal neighbourhood of filter_time time units. Useful to filter noisy recordings
    where events occur isolated in time.

    Parameters:
        filter_time (float): maximum temporal distance to next event, otherwise dropped.
                    Lower values will mean higher constraints, therefore less events.
    """

    def __init__(self, filter_time: float = 10000):
        self.filter_time = filter_time

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.denoise_numpy(
            events=events,
            sensor_size=sensor_size,
            ordering=ordering,
            filter_time=self.filter_time,
        )
        return events, images, sensor_size


class DropEvent:
    """Randomly drops events with drop_probability.

    Parameters:
        drop_probability (float): probability of dropping out event.
        random_drop_probability (bool): randomize the dropout probability
                                 between 0 and drop_probability.
    """

    def __init__(
        self, drop_probability: float = 0.5, random_drop_probability: bool = False
    ):
        self.drop_probability = drop_probability
        self.random_drop_probability = random_drop_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.drop_event_numpy(
            events, self.drop_probability, self.random_drop_probability
        )
        return events, images, sensor_size


class DropPixel:
    """Drops events for individual pixels. If the locations of pixels to be dropped is known, a
    list of x/y coordinates can be passed directly. Alternatively, a cutoff frequency for each pixel can be defined
    above which pixels will be deactivated completely. This prevents so-called _hot pixels_ which fire constantly
    (e.g. due to faulty hardware).

    Parameters:
        coordinates: list of (x,y) coordinates for which all events will be deleted.
    """

    def __init__(self, coordinates=None, hot_pixel_frequency=None):
        self.coordinates = coordinates
        self.hot_pixel_frequency = hot_pixel_frequency

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        if self.hot_pixel_frequency:
            self.coordinates = functional.identify_hot_pixel(
                events=events,
                sensor_size=sensor_size,
                ordering=ordering,
                hot_pixel_frequency=self.hot_pixel_frequency,
            )

        print(f"Filtered {len(self.coordinates)} hot pixels.")

        events = functional.drop_pixel_numpy(
            events=events, ordering=ordering, coordinates=self.coordinates,
        )
        return events, images, sensor_size


class Downsample:
    """Multiplies timestamps and spatial pixel coordinates with separate factors.
    Useful when the native temporal and/or spatial resolution of the original sensor is too
    high for downstream processing, notably when converting to dense representations of some sort.
    This transform does not drop any events.

    Parameters:
        time_factor (float): value to multiply timestamps with. Default is 0.001.
        spatial_factor (float): value to multiply pixel coordinates with. Default is 1.
    """

    def __init__(self, time_factor: float = 1e-3, spatial_factor: float = 1):
        self.time_factor = time_factor
        self.spatial_factor = spatial_factor

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.time_skew_numpy(
            events, ordering, coefficient=self.time_factor
        )
        events, sensor_size = functional.spatial_resize_numpy(
            events, sensor_size, ordering, spatial_factor=self.spatial_factor, integer_coordinates=True
        )
        return events, images, sensor_size


class FlipLR:
    """Flips events and images in x. Pixels map as:

        x' = width - x

    Parameters:
        flip_probability (float): probability of performing the flip
    """

    def __init__(self, flip_probability: float = 0.5):
        self.flip_probability = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.flip_lr_numpy(
            events=events,
            sensor_size=sensor_size,
            ordering=ordering,
            images=images,
            multi_image=multi_image,
            flip_probability=self.flip_probability,
        )


class FlipPolarity:
    """Flips polarity of individual events with flip_probability.
    Changes polarities 1 to -1 and polarities [-1, 0] to 1

    Parameters:
        flip_probability (float): probability of flipping individual event polarities
    """

    def __init__(self, flip_probability: float = 0.5):
        self.flip_probability = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.flip_polarity_numpy(
            events=events, ordering=ordering, flip_probability=self.flip_probability
        )
        return events, images, sensor_size


class FlipUD:
    """
    Flips events and images in y. Pixels map as:

        y' = height - y

    Parameters:
        flip_probability (float): probability of performing the flip
    """

    def __init__(self, flip_probability: float = 0.5):
        self.flip_probability = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.flip_ud_numpy(
            events=events,
            sensor_size=sensor_size,
            ordering=ordering,
            images=images,
            multi_image=multi_image,
            flip_probability=self.flip_probability,
        )


class NumpyAsType:
    """
    Change dtype of numpy ndarray to custom dtype.
    
    Parameters: 
        dtype: data type that the array should be cast to
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = events.astype(self.dtype)

        if images is not None:
            images = images.astype(self.dtype)

        return events, images, sensor_size


class RefractoryPeriod:
    """Sets a refractory period for each pixel, during which events will be
    ignored/discarded. We keep events if:

        .. math::
            t_n - t_{n-1} > t_{refrac}
    
    for each pixel.

    Parameters:
        refractory_period (float): refractory period for each pixel in time unit
    """

    def __init__(self, refractory_period: float):
        self.refractory_period = refractory_period

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.refractory_period_numpy(
            events, sensor_size, ordering, self.refractory_period
        )
        return events, images, sensor_size


class SpatialJitter:
    """Changes position for each pixel by drawing samples from a multivariate
    Gaussian distribution with the following properties:

        mean = [x,y]
        covariance matrix = [[variance_x, sigma_x_y],[sigma_x_y, variance_y]]

    Jittered events that lie outside the focal plane will be dropped if clip_outliers is True.

    Parameters:
        variance_x (float): squared sigma value for the distribution in the x direction
        variance_y (float): squared sigma value for the distribution in the y direction
        sigma_x_y (float): changes skewness of distribution, only change if you want shifts along diagonal axis.
        integer_jitter (bool): when True, x and y coordinates will be shifted by integer rather values instead of floats.
        clip_outliers (bool): when True, events that have been jittered outside the sensor size will be dropped.
    """

    def __init__(
        self,
        variance_x: float = 1,
        variance_y: float = 1,
        sigma_x_y: float = 0,
        integer_jitter: bool = False,
        clip_outliers: bool = False,
    ):
        self.variance_x = variance_x
        self.variance_y = variance_y
        self.sigma_x_y = sigma_x_y
        self.integer_jitter = integer_jitter
        self.clip_outliers = clip_outliers

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.spatial_jitter_numpy(
            events=events,
            sensor_size=sensor_size,
            ordering=ordering,
            variance_x=self.variance_x,
            variance_y=self.variance_y,
            sigma_x_y=self.sigma_x_y,
            integer_jitter=self.integer_jitter,
            clip_outliers=self.clip_outliers,
        )
        return events, images, sensor_size


class SpatioTemporalTransform:
    """Transform all events spatial and temporal locations based on
    given spatial transform matrix and temporal transform vector.

    Parameters:
        spatial_transform: 3x3 matrix which can be used to perform rigid (translation and rotation),
                           non-rigid (scaling and shearing), and non-affine transformations. Generic to user input.
        temporal_transform: scale time between events and offset temporal location based on 2 member vector.
                            Used as arguments to time_skew method.
        roll: boolean input to determine if transformed events will be translated across sensor boundaries (True).
              Otherwise, events will be clipped at sensor boundaries.
    """

    def __init__(self, spatial_transform, temporal_transform, roll=False):
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.roll = roll

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.st_transform(
            events,
            sensor_size,
            ordering,
            self.spatial_transform,
            self.temporal_transform,
            self.roll,
        )
        return events, images, sensor_size
    
    
class TimeAlignment:
    """Shifts the timestamps to set the first event of all recordings of the dataset to zero.
    """

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        assert "t" in ordering
        t_index = ordering.index("t")
        events[:,t_index] -= min(events[:, t_index])
        return events, images, sensor_size


class TimeJitter:
    """Changes timestamp for each event by drawing samples from a Gaussian
    distribution and adding them to each timestamp.

    Parameters:
        std (float): change the standard deviation of the time jitter
        integer_jitter (bool): will round the jitter that is added to timestamps
        clip_negative (bool): drops events that have negative timestamps
        sort_timestamps (bool): sort the events by timestamps
    """

    def __init__(
        self,
        std: float = 1,
        integer_jitter: bool = False,
        clip_negative: bool = False,
        sort_timestamps: bool = False,
    ):
        self.std = std
        self.integer_jitter = integer_jitter
        self.clip_negative = clip_negative
        self.sort_timestamps = sort_timestamps

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.time_jitter_numpy(
            events,
            ordering,
            self.std,
            self.integer_jitter,
            self.clip_negative,
            self.sort_timestamps,
        )
        return events, images, sensor_size


class TimeReversal:
    """Temporal flip is defined as:

        .. math::
           t_i' = max(t) - t_i

           p_i' = -1 * p_i

    Parameters:
        flip_probability (float): probability of performing the flip
    """

    def __init__(self, flip_probability: float = 0.5):
        self.flip_probability = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.time_reversal_numpy(
            events=events,
            sensor_size=sensor_size,
            ordering=ordering,
            images=images,
            multi_image=multi_image,
            flip_probability=self.flip_probability,
        )


class TimeSkew:
    """Skew all event timestamps according to a linear transform,
    potentially sampled from a distribution of acceptable functions.

    Parameters:
        coefficient: a real-valued multiplier applied to the timestamps of the events.
                     E.g. a coefficient of 2.0 will double the effective delay between any
                     pair of events.
        offset: value by which the timestamps will be shifted after multiplication by
                the coefficient. Negative offsets are permissible but may result in
                in an exception if timestamps are shifted below 0.
        integer_time: flag that specifies if timestamps should be rounded to
                             nearest integer after skewing.
    """

    def __init__(
        self, coefficient: float, offset: float = 0, integer_time: bool = False
    ):
        self.coefficient = coefficient
        self.offset = offset
        self.integer_time = integer_time

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.time_skew_numpy(
            events, ordering, self.coefficient, self.offset, self.integer_time
        )
        return events, images, sensor_size


class UniformNoise:
    """
    Introduces a fixed number of noise depending on sensor size and noise
    density factor, uniformly distributed across the focal plane and in time.

    Parameters:
        scaling_factor_to_micro_sec: this is a scaling factor to get to micro
                                     seconds from the time resolution used in the event stream,
                                     as the noise time resolution is fixed to 1 micro second.
        noise_density: A noise density of 1 will mean one noise event for every
                       pixel of the sensor size for every micro second.
    """

    def __init__(
        self, scaling_factor_to_micro_sec: float = 1, noise_density: float = 1e-8
    ):
        self.scaling_factor_to_micro_sec = scaling_factor_to_micro_sec
        self.noise_density = noise_density

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.uniform_noise_numpy(
            events,
            sensor_size,
            ordering,
            self.scaling_factor_to_micro_sec,
            self.noise_density,
        )
        return events, images, sensor_size


class ToAveragedTimesurface:
    """Representation that creates averaged timesurfaces for each event for one recording. Taken from the paper
    Sironi et al. 2018, HATS: Histograms of averaged time surfaces for robust event-based object classification
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf

    Parameters:
        cell_size (int): size of each square in the grid
        surface_size (int): has to be odd
        time_window (float): how far back to look for past events for the time averaging
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.
    """

    def __init__(
        self,
        cell_size=10,
        surface_size=7,
        temporal_window=5e5,
        tau=5e3,
        decay="lin",
        merge_polarities=False,
    ):
        assert surface_size % 2 == 1
        self.cell_size = cell_size
        self.surface_size = surface_size
        self.temporal_window = temporal_window
        self.tau = tau
        self.decay = decay
        self.merge_polarities = merge_polarities

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        surfaces = functional.to_averaged_timesurface(
            events,
            sensor_size,
            ordering,
            self.cell_size,
            self.surface_size,
            self.temporal_window,
            self.tau,
            self.decay,
            self.merge_polarities,
        )
        return surfaces, images


class ToFrame:
    """Accumulate events to frames by slicing along constant time (time_window),
    constant number of events (spike_count) or constant number of frames (n_time_bins / n_event_bins).
    You can set one of the first 4 parameters to choose the slicing method. Depending on which method you choose,
    overlap will assume different functionality, whether that might be temporal overlap, number of events
    or fraction of a bin. As a rule of thumb, here are some considerations if you are unsure which slicing
    method to choose:

    * If your recordings are of roughly the same length, a safe option is to set time_window. Bare in mind
      that the number of events can vary greatly from slice to slice, but will give you some consistency when
      training RNNs or other algorithms that have time steps.

    * If your recordings have roughly the same amount of activity / number of events and you are more interested
      in the spatial composition, then setting spike_count will give you frames that are visually more consistent.

    * The previous time_window and spike_count methods will likely result in a different amount of frames for each
      recording. If your training method benefits from consistent number of frames across a dataset (for easier
      batching for example), or you want a parameter that is easier to set than the exact window length or number
      of events per slice, consider fixing the number of frames by setting n_time_bins or n_event_bins. The two
      methods slightly differ with respect to how the slices are distributed across the recording. You can define
      an overlap between 0 and 1 to provide some robustness.

    Parameters:
        time_window (float): time window length for one frame. Use the same time unit as timestamps in the event recordings.
                             Good if you want temporal consistency in your training, bad if you need some visual consistency
                             for every frame if the recording's activity is not consistent.
        spike_count (int): number of events per frame. Good for training CNNs which do not care about temporal consistency.
        n_time_bins (int): fixed number of frames, sliced along time axis. Good for generating a pre-determined number of
                           frames which might help with batching.
        n_event_bins (int): fixed number of frames, sliced along number of events in the recording. Good for generating a
                            pre-determined number of frames which might help with batching.
        overlap (float): overlap between frames defined either in time units, number of events or number of bins between 0 and 1.
        include_incomplete (bool): if True, includes overhang slice when time_window or spike_count is specified.
                                   Not valid for bin_count methods.
        merge_polarities (bool): if True, merge polarity channels to a single channel.
    """

    def __init__(
        self,
        time_window=None,
        spike_count=None,
        n_time_bins=None,
        n_event_bins=None,
        overlap=0.0,
        include_incomplete=False,
        merge_polarities=False,
    ):
        self.time_window = time_window
        self.spike_count = spike_count
        self.n_time_bins = n_time_bins
        self.n_event_bins = n_event_bins
        self.overlap = overlap
        self.include_incomplete = include_incomplete
        self.merge_polarities = merge_polarities

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        frames = functional.to_frame_numpy(
            events=events,
            sensor_size=sensor_size,
            ordering=ordering,
            time_window=self.time_window,
            spike_count=self.spike_count,
            n_time_bins=self.n_time_bins,
            n_event_bins=self.n_event_bins,
            overlap=self.overlap,
            include_incomplete=self.include_incomplete,
            merge_polarities=self.merge_polarities,
        )
        return frames, images, sensor_size


class ToSparseTensor:
    """Turn event array (N,E) into sparse Tensor (B,T,W,H) if E is 4 (mostly event camera recordings),
    otherwise into sparse tensor (B,T,W) for mostly audio recordings."""

    def __init__(self, type="pytorch", merge_polarities=False):
        self.merge_polarities = merge_polarities
        self.type = type

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        if self.type == "pytorch" or "pt" or "torch":
            tensor = functional.to_sparse_tensor_pytorch(
                events=events,
                sensor_size=sensor_size,
                ordering=ordering,
                merge_polarities=self.merge_polarities,
            )
        elif self.type == "tensorflow" or "tf":
            tensor = functional.to_sparse_tensor_tensorflow(
                events=events,
                sensor_size=sensor_size,
                ordering=ordering,
                merge_polarities=self.merge_polarities,
            )
        else:
            raise NotImplementedError
        return tensor, images, sensor_size


class ToTimesurface:
    """Representation that creates timesurfaces for each event in the recording. Modeled after the paper
    Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern recognition
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476

    Parameters:
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.
    """

    def __init__(
        self, surface_dimensions=(7, 7), tau=5e3, decay="lin", merge_polarities=False
    ):
        if surface_dimensions:
            assert len(surface_dimensions) == 2
            assert surface_dimensions[0] % 2 == 1 and surface_dimensions[1] % 2 == 1
        else:
            pass
        self.surface_dimensions = surface_dimensions
        self.tau = tau
        self.decay = decay
        self.merge_polarities = merge_polarities

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        surfaces = functional.to_timesurface_numpy(
            events,
            sensor_size,
            ordering,
            self.surface_dimensions,
            self.tau,
            self.decay,
            self.merge_polarities,
        )
        return surfaces, images, sensor_size


class ToVoxelGrid:
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events."""

    def __init__(self, n_time_bins):
        self.n_time_bins = n_time_bins

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        volume = functional.to_voxel_grid_numpy(
            events, sensor_size, ordering, self.n_time_bins
        )
        return volume, images, sensor_size


class Repeat:
    """Copies target n times. Useful to transform sample labels into sequences."""

    def __init__(self, repetitions):
        self.repetitions = repetitions

    def __call__(self, target):
        return np.tile(np.expand_dims(target, 0), [self.repetitions, 1])


class ToOneHotEncoding:
    """Transforms one or more targets into a one hot encoding scheme."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, target):
        return np.eye(self.n_classes)[target]
