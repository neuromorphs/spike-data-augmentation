import os
import warnings
from typing import Callable, List, Optional, Union

import h5py
import numpy as np

from tonic.dataset import Dataset
from tonic.download_utils import download_and_extract_archive, download_url, list_files
from tonic.io import make_structured_array


class DSEC(Dataset):
    """`DSEC <https://dsec.ifi.uzh.ch/>`_

    This is a fairly large dataset, so in order to save some disk space, event and image zips
    are deleted after extraction. If your download gets interrupted and you are left with a
    corrupted file on disk, Tonic will not be able to detect that and just proceed to download
    files that are not yet on disk. If you experience issues loading a particular recording,
    delete that folder manually and Tonic will re-download it the next time.
    Optical flow targets are not available for every recording, so if you select optical flow targets,
    only a subset of 18 training recordings will be selected.

    .. note:: To be able to read this dataset, you will need `hdf5plugin`, `PIL` and `imageio` packages installed.

    Parameters:
        save_to (str): Location to save files to on disk.
        split (str): Can be 'train', 'test' or a selection of individual recordings such as 'interlaken_00_c'
                     or ['thun_00_a', 'zurich_city_00_a']. Cannot mix across train/test.
        data_selection (str): Select which data to load per sample. Can be 'events_left', 'events_right',
                             'images_rectified_left', 'images_rectified_right', 'image_timestamps' or
                             any combination thereof in a list.
        target_selection (str, optional): Select which targets to load per sample. Can be 'disparity_events',
                                          'disparity_images', 'disparity_timestamps',
                                          'optical_flow_forward_event', 'optical_flow_forward_timestamps',
                                          'optical_flow_backward_event', 'optical_flow_backward_timestamps'
                                          or a combination thereof in a list. Note that optical flow targets
                                          are not available for every recording.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    base_url = "https://download.ifi.uzh.ch/rpg/DSEC/"

    # boolean flag indicates optical flow availability
    recordings = {
        "train": {
            "interlaken_00_c": False,
            "interlaken_00_d": False,
            "interlaken_00_e": False,
            "interlaken_00_f": False,
            "interlaken_00_g": False,
            "thun_00_a": True,
            "zurich_city_00_a": False,
            "zurich_city_00_b": False,
            "zurich_city_01_a": True,
            "zurich_city_01_b": False,
            "zurich_city_01_c": False,
            "zurich_city_01_d": False,
            "zurich_city_01_e": False,
            "zurich_city_01_f": False,
            "zurich_city_02_a": True,
            "zurich_city_02_b": False,
            "zurich_city_02_c": True,
            "zurich_city_02_d": True,
            "zurich_city_02_e": True,
            "zurich_city_03_a": True,
            "zurich_city_04_a": False,
            "zurich_city_04_b": False,
            "zurich_city_04_c": False,
            "zurich_city_04_d": False,
            "zurich_city_04_e": False,
            "zurich_city_04_f": False,
            "zurich_city_05_a": True,
            "zurich_city_05_b": True,
            "zurich_city_06_a": True,
            "zurich_city_07_a": True,
            "zurich_city_08_a": True,
            "zurich_city_09_a": True,
            "zurich_city_09_b": False,
            "zurich_city_09_c": False,
            "zurich_city_09_d": False,
            "zurich_city_09_e": False,
            "zurich_city_10_a": True,
            "zurich_city_10_b": True,
            "zurich_city_11_a": True,
            "zurich_city_11_b": True,
            "zurich_city_11_c": True,
        },
        "test": [
            "thun_01_a",
            "thun_01_b",
            "interlaken_00_a",
            "interlaken_00_b",
            "interlaken_01_a",
            "zurich_city_12_a",
            "zurich_city_13_a",
            "zurich_city_13_b",
            "zurich_city_14_a",
            "zurich_city_14_b",
            "zurich_city_14_c",
            "zurich_city_15_a",
        ],
    }

    # that's a combination of the different data available, their
    # extension when downloaded and their extension when extracted
    data_names = {
        "events_left": [".zip", ".h5"],
        "events_right": [".zip", ".h5"],
        "image_timestamps": [".txt", ".txt"],
        "image_exposure_timestamps_left": [".txt", ".txt"],
        "image_exposure_timestamps_right": [".txt", ".txt"],
        "images_rectified_left": [".zip", ".png"],
        "images_rectified_right": [".zip", ".png"],
    }

    target_names = {
        "disparity_event": [".zip", ".png"],
        "disparity_image": [".zip", ".png"],
        "disparity_timestamps": [".txt", ".txt"],
        "optical_flow_forward_event": [".zip", ".png"],
        "optical_flow_forward_timestamps": [".txt", ".txt"],
        "optical_flow_backward_event": [".zip", ".png"],
        "optical_flow_backward_timestamps": [".txt", ".txt"],
    }

    sensor_size = (640, 480, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        split: Union[str, List[str]],
        data_selection: Union[str, List[str]],
        target_selection: Optional[Union[str, List[str]]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        import imageio

        # imageio.plugins.freeimage.download()

        if split == "train":
            self.recording_selection = self.recordings[split].keys()
            self.train_or_test = split

        elif split == "test":
            self.recording_selection = self.recordings[split]
            self.train_or_test = split

        else:
            if not isinstance(split, list):
                split = [split]

            for recording in split:
                if (
                    recording not in self.recordings["train"]
                    and recording not in self.recordings["test"]
                ):
                    raise RuntimeError(
                        f"Recording {recording} is neither in train nor in test set."
                    )
            self.recording_selection = split
            if all([recording in self.recordings["train"] for recording in split]):
                self.train_or_test = "train"
            elif all([recording in self.recordings["test"] for recording in split]):
                self.train_or_test = "test"
            else:
                raise RuntimeError("Cannot mix across train/test split.")
        self.train = self.train_or_test == "train"

        if isinstance(data_selection, str):
            data_selection = [data_selection]
        elif data_selection is None:
            data_selection = []
        self.data_selection = data_selection

        for data_piece in data_selection:
            if data_piece not in self.data_names.keys():
                raise RuntimeError(
                    f"Selection {data_piece} not available. Please select from the following options: {self.data_names.keys()}."
                )

        if isinstance(target_selection, str):
            target_selection = [target_selection]
        elif target_selection is None:
            target_selection = []
        self.target_selection = target_selection

        if not self.train and len(target_selection) > 0:
            raise Exception(
                "You wanted targets for the test set but they are not available."
            )

        for data_piece in target_selection:
            if data_piece not in self.target_names.keys():
                raise RuntimeError(
                    f"Selection {data_piece} not available. Please select from the following options: {self.target_names.keys()}."
                )

        # only take those recordings that have optical flow ground truth
        if any(["optical_flow" in selection for selection in target_selection]):
            deselect = [
                name
                for name in self.recording_selection
                if not self.recordings["train"][name]
            ]
            if len(deselect) > 0:
                warnings.warn(
                    f"Since you asked for optical flow targets, the following recordings without optical flow ground truth are dropped: {deselect}."
                )
                self.recording_selection = [
                    name
                    for name in self.recording_selection
                    if self.recordings["train"][name]
                ]

        self._check_exists(data_selection + target_selection)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (data, target) where data is another tuple of data_selction and target
            a tuple of target_selection if train=True.
        """
        import hdf5plugin  # necessary to read event files
        import imageio  # necessary to read optical flow pngs
        from PIL import Image  # necessary to read images

        recording = self.recording_selection[index]
        base_folder = os.path.join(self.location_on_system, recording)

        data_tuple = []
        for data_name in self.data_selection:
            full_base_folder = os.path.join(base_folder, data_name)
            if data_name in ["events_left", "events_right"]:
                with h5py.File(full_base_folder + "/events.h5", "r") as file:
                    data = {}
                    data[data_name] = make_structured_array(
                        file["events"]["x"][()],
                        file["events"]["y"][()],
                        file["events"]["t"][()],
                        file["events"]["p"][()],
                        dtype=self.dtype,
                    )
                    data[data_name]["t"] += file["t_offset"][()]
                    data["ms_to_idx"] = file["ms_to_idx"][()]

            elif "images" in data_name:
                images_rectified_filenames = sorted(
                    list_files(full_base_folder, ".png", prefix=True)
                )
                data = np.stack(
                    [np.array(Image.open(file)) for file in images_rectified_filenames]
                )

            elif data_name == "image_timestamps":
                with open(full_base_folder + f"/{recording}_image_timestamps.txt") as f:
                    data = np.array([int(line) for line in f.readlines()])
            data_tuple.append(data)

        if self.transform is not None:
            data_tuple = self.transform(data_tuple)

        target_tuple = []
        for target_name in self.target_selection:
            full_base_folder = os.path.join(base_folder, target_name)
            if target_name in [
                "disparity_event",
                "disparity_image",
            ]:
                png_filenames = sorted(
                    list_files(full_base_folder, ".png", prefix=True)
                )
                target = np.stack(
                    [np.array(Image.open(file)) for file in png_filenames]
                )

            elif target_name in [
                "optical_flow_forward_event",
                "optical_flow_backward_event",
            ]:
                png_filenames = sorted(
                    list_files(full_base_folder, ".png", prefix=True)
                )
                target = np.array(
                    [imageio.v2.imread(file, format="PNG-FI") for file in png_filenames]
                ).astype(float)
                target[:, :, :, :2] -= 2**15
                target[:, :, :, :2] /= 128

            elif target_name == "disparity_timestamps":
                with open(full_base_folder + f"/{recording}_{target_name}.txt") as f:
                    target = np.array([int(line) for line in f.readlines()])

            elif target_name in [
                "optical_flow_forward_timestamps",
                "optical_flow_backward_timestamps",
            ]:
                with open(full_base_folder + f"/{recording}_{target_name}.txt") as f:
                    lines = f.readlines()
                    lines = lines[1:]  # first line is a comment
                    # first number is start timestamp, second number is stop timestamp
                    number_strs = [line.split(", ") for line in lines]
                    target = np.array(
                        [(int(start), int(stop)) for start, stop in number_strs]
                    )

            target_tuple.append(target)

        if self.target_transform is not None:
            target_tuple = self.target_transform(target_tuple)

        if self.transforms is not None:
            data_tuple, target_tuple = self.transforms(data_tuple, target_tuple)
        return data_tuple, target_tuple

    def __len__(self):
        return len(self.recording_selection)

    def _check_exists(self, data_selection: List):
        all_names = {**self.data_names, **self.target_names}
        for recording in self.recording_selection:
            for data_name in data_selection:
                file_folder = os.path.join(
                    self.location_on_system, recording, data_name
                )
                os.makedirs(file_folder, exist_ok=True)
                extension, extracted_file_extension = all_names[data_name]
                file_name = f"{recording}_{data_name + extension}"

                if any(
                    file.endswith(extracted_file_extension)
                    for file in os.listdir(file_folder)
                ):
                    continue

                url = f"{self.base_url}{self.train_or_test}/{recording}/{file_name}"
                if extension == ".zip":
                    download_and_extract_archive(url, file_folder)
                    os.remove(os.path.join(file_folder, file_name))
                else:
                    download_url(url, file_folder)
