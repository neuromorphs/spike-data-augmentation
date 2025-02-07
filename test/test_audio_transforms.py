import numpy as np
import pytest
from tonic.audio_transforms import (AddNoise, AmplitudeScale, Bin, FixLength,
                                    LinearButterFilterBank,
                                    MelButterFilterBank, RobustAmplitudeScale,
                                    SwapAxes)


class DummyNoiseDataset:
    """Dummy dataset for generating random noise signals of varying lengths."""

    def __len__(self) -> int:
        return 1000

    def __getitem__(self, item: int) -> tuple[np.ndarray, int]:
        sig_len = np.random.randint(12000, 20000)
        return np.random.random((1, sig_len)), 0


def test_standardize_data_length():
    """Test FixLength transform for both extending and truncating data."""
    sdl = FixLength(100, 1)

    # Data is longer
    data_long = np.ones((1, 120))
    assert sdl(data_long).shape == (1, 100)

    # Data is shorter
    data_short = np.ones((1, 80))
    assert sdl(data_short).shape == (1, 100)


def test_bin():
    """Test Bin transform for re-sampling data frequency."""
    bin_transform = Bin(orig_freq=16000, new_freq=100, axis=1)

    data = np.random.random((1, 8 * 16000))
    data_binned = bin_transform(data)

    assert data_binned.shape == (1, 8 * 100)
    assert data.sum() == pytest.approx(data_binned.sum(), 1e-7)


def test_linear_butter_filter_bank():
    """Test LinearButterFilterBank for filtering data."""
    fb = LinearButterFilterBank(
        order=2, low_freq=100, sampling_freq=16000, num_filters=16
    )
    data = np.random.random((1, 16000))

    filter_out = fb(data)
    assert filter_out.shape == (16, 16000)


def test_mel_butter_filter_bank():
    """Test MelButterFilterBank for filtering data using Mel scale."""
    fb = MelButterFilterBank(order=2, low_freq=100, sampling_freq=16000, num_filters=16)
    data = np.random.random((1, 16000))

    filter_out = fb(data)
    assert filter_out.shape == (16, 16000)


def test_add_noise():
    """Test AddNoise transform for adding noise to a signal."""
    data = np.sin(np.linspace(0, 2 * np.pi, 16000))[None, ...]
    noise_dataset = DummyNoiseDataset()
    add_noise = AddNoise(noise_dataset, 10, normed=True)

    signal = add_noise(data)
    assert signal.shape == (1, 16000)


def test_swap_axes():
    """Test SwapAxes transform for swapping axes of the data."""
    data = np.random.rand(1, 16000)
    swap_ax = SwapAxes(ax1=0, ax2=1)
    swaped = swap_ax(data)

    assert swaped.shape == (16000, 1)


def test_amplitude_scale():
    """Test AmplitudeScale transform for scaling amplitude of the data."""
    data = np.random.rand(1, 16000)
    max_amps = np.random.rand(10)

    for amp in max_amps:
        amp_scale = AmplitudeScale(max_amplitude=amp)
        transformed = amp_scale(data)
        assert transformed.max() == amp


def test_robust_amplitude_scale():
    """Test RobustAmplitudeScale transform for scaling amplitude considering outliers."""
    data = np.random.rand(1, 16000)
    max_amps = np.random.rand(10)
    percent = 0.01

    for amp in max_amps:
        robust_amp_scale = RobustAmplitudeScale(
            max_robust_amplitude=amp, outlier_percent=percent
        )
        transformed = robust_amp_scale(data)
        sorted_transformed = np.sort(np.abs(transformed.ravel()))
        non_outlier = sorted_transformed[: int(len(sorted_transformed) * (1 - percent))]

        assert np.all(non_outlier <= amp)
