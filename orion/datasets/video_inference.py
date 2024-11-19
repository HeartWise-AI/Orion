"""orion-Dynamic Dataset."""
import collections
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import skimage.draw
import torch.utils.data
from torchvision.transforms import v2

import orion

dir2 = os.path.abspath("/volume/Oread/orion")
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)


class Video_inference(torch.utils.data.Dataset):
    """orion Dataset.
    Args:
        root (string): Root directory of dataset (defaults to `orion.config.DATA_DIR`)
        split (string): One of {"train", "val", "test", "inference"}
        target_label (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filenam e of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(
        self,
        root="",
        data_filename=None,
        target_label=None,
        split="train",
        datapoint_loc_label="FileName2",
        resize=224,
        mean=0.0,
        std=1.0,
        length=32,
        period=1,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        video_transforms=None,
        apply_mask=False,
        num_classes=None,
        target_transform=None,
        external_test_location=None,
    ) -> None:
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split
        self.mean = format_mean_std(mean)
        self.std = format_mean_std(std)
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.video_transforms = video_transforms
        self.apply_mask = apply_mask
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.resize = resize

        self.fnames, self.outcome = [], []

        df_dataset = pd.read_csv(
            os.path.join(self.folder, self.filename), sep="α", engine="python"
        )
        self.header = list(df_dataset.columns)

        if len(self.header) == 1:
            raise ValueError(
                "Header was not split properly. Please ensure the file uses 'α' (alpha) as the delimiter."
            )
        # Add this line to store the DataFrame's header
        filenameIndex = list(df_dataset.columns).index(self.datapoint_loc_label)

        splitIndex = list(df_dataset.columns).index("Split")

        for i, row in df_dataset.iterrows():
            try:
                fileName = os.path.join(self.folder, row.iloc[filenameIndex])
                if pd.isna(fileName) or fileName == "":
                    raise ValueError("Filename is not defined or empty.")
            except ValueError as ve:
                print(ve)

            fileMode = str(row.iloc[splitIndex]).lower()
            if split in ["all", fileMode]:
                if os.path.exists(fileName):
                    self.fnames.append(fileName)
                    if (target_index is not None) and (not pd.isna(row.iloc[target_index])):
                        self.outcome.append(row.iloc[target_index])
                else:
                    raise FileNotFoundError(f"The file {fileName} does not exist.")

        if len(self.fnames) == 0:
            raise ValueError(f"No files found for split {split}")

        self.frames = collections.defaultdict(list)
        # pdb.set_trace()
        self.trace = collections.defaultdict(_defaultdict_of_lists)

    #         # define weights for weighted sampling
    #         labels = np.array([self.outcome[ind][target_index] for ind in range(len(self.outcome))], dtype=int)
    #         # binary weights length == 2
    #         weights = 1 - (np.bincount(labels) / len(labels))
    #         self.weight_list = np.zeros(len(labels))

    #         for label in range(len(weights)):
    #             weight = weights[label]
    #             self.weight_list[np.where(labels == label)] = weight

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "external_test":
            video_fname = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "clinical_test":
            video_fname = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_fname = self.fnames[index]

        # Load video into np.array
        # video = orion.utils.loadvideo(video).astype(np.float32)
        # transforms
        video = orion.utils.loadvideo(video_fname).astype(np.float32)

        if self.apply_mask:
            path = video_fname.rsplit("/", 2)
            mask_filename = path[0] + "/mask/" + path[2]
            mask_filename = mask_filename.split(".avi")[0] + ".npy"

            mask = np.load(mask_filename).transpose(2, 0, 1)

            # fix mask shapes
            length = video.shape[2]
            if mask.shape[1] < length:
                mask = np.pad(mask, [(0, 0), (length - mask.shape[1], 0), (0, 0)])
            if mask.shape[2] < length:
                mask = np.pad(mask, [(0, 0), (0, 0), (length - mask.shape[2], 0)])
            if mask.shape[1] > length:
                mask = mask[:, :length, :]
            if mask.shape[2] > length:
                mask = mask[:, :, :length]

            for ind in range(video.shape[0]):
                video[ind, :, :, :] = video[ind, :, :, :] * mask

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        video = torch.from_numpy(video)
        if self.resize is not None:
            video = v2.Resize((self.resize, self.resize), antialias=None)(video)
        #       apply normalization before augmentaiton per riselab and pytorchvideo
        video = v2.Normalize(self.mean, self.std)(video)
        #         pdb.set_trace()

        if self.video_transforms is not None:
            transforms = v2.RandomApply(torch.nn.ModuleList(self.video_transforms), p=0.5)
            scripted_transforms = torch.jit.script(transforms)
            video = scripted_transforms(video)

        #             transform = v2.Compose([v2.RandomErasing()])
        #             video = transform(video)

        video = video.permute(1, 0, 2, 3)
        video = video.numpy()

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate(
                (video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1
            )
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            if self.split == "train":
                start = np.random.choice(f - (length - 1) * self.period, self.clips)
            else:
                start = np.array([0])

        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.split == "inference" or self.target_label is None:
            target = None
        else:
            target = self.outcome[index]

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[
                :, :, self.pad : -self.pad, self.pad : -self.pad
            ] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i : (i + h), j : (j + w)]

            return video, self.fnames[index]
        
        if target is not None:
            return video, target, self.fnames[index]
        else:
            return video, self.fnames[index]        

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class Video_Multi_inference(torch.utils.data.Dataset):
    """orion Dataset for multiple video inference."""

    def __init__(
        self,
        root="../../data/",
        data_filename=None,
        split="inference",
        target_label=None,
        datapoint_loc_label="FileName",
        resize=224,
        view_count=2,
        mean=0.0,
        std=1.0,
        length=32,
        period=1,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        video_transforms=None,
        apply_mask=False,
        rand_augment=False,
        num_classes=None,
        target_transform=None,
        external_test_location=None,
        weighted_sampling=False,
        debug=False,
        normalize=True,
    ) -> None:
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split
        self.mean = format_mean_std(mean)
        self.std = format_mean_std(std)
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.video_transforms = video_transforms
        self.apply_mask = apply_mask
        self.rand_augment = rand_augment
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.resize = resize
        self.view_count = view_count
        self.debug = debug
        self.weighted_sampling = weighted_sampling
        self.normalize = normalize

        self.fnames = []

        df_dataset = pd.read_csv(
            os.path.join(self.folder, self.filename), sep="α", engine="python"
        )

        filenameIndex = df_dataset.columns.get_loc(self.datapoint_loc_label + str(0))
        splitIndex = df_dataset.columns.get_loc("Split")
        for _, row in df_dataset.iterrows():
            view_count = int(self.view_count)
            filenameIndex = [
                df_dataset.columns.get_loc(self.datapoint_loc_label + str(i))
                for i in range(view_count)
            ]
            fileMode = row.iloc[splitIndex].lower()
            fileName = [row.iloc[i] for i in filenameIndex]

            file_vids = [
                i for i in fileName if self.split in ["all", fileMode] and os.path.exists(i)
            ]
            if len(file_vids) > 1:
                self.fnames.append(file_vids)

        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

    def make_video(self, video, count, index):
        video = orion.utils.loadvideo(video).astype(np.float32)

        if self.apply_mask:
            # Apply mask logic here (similar to Video_Multi)
            pass

        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        video = torch.from_numpy(video)

        if self.resize is not None:
            video = v2.Resize((self.resize, self.resize), antialias=True)(video)

        if self.normalize:
            if hasattr(self, "mean") and hasattr(self, "std"):
                video = v2.Normalize(self.mean, self.std)(video)

        video = video.permute(1, 0, 2, 3)
        video = video.numpy()

        c, f, h, w = video.shape
        length = f // self.period if self.length is None else self.length
        if self.max_length is not None:
            length = min(length, self.max_length)

        if f < length * self.period:
            video = np.concatenate(
                (video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1
            )
            c, f, h, w = video.shape

        if self.clips == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        video = video[0] if self.clips == 1 else np.stack(video)

        if self.pad is not None:
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad : -self.pad, self.pad : -self.pad] = video
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i : (i + h), j : (j + w)]

        return video

    def __getitem__(self, index):
        vids_stack = []
        filename_stack = []
        filenames = self.fnames[index]

        for count, i in enumerate(filenames):
            video = self.make_video(i, count, index)
            vids_stack.append(video)
            filename_stack.append(i)

        return vids_stack, filename_stack

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


def format_mean_std(input_value):
    """
    Formats the mean or std value to a list of floats.

    Args:
        input_value (str, list, np.array): The input mean/std value.

    Returns:
        list: A list of floats.

    Raises:
        ValueError: If the input cannot be converted to a list of floats.
    """

    if input_value is 1.0 or input_value is 0.0:
        print("Mean/STD value is not defined")
        return input_value

    # Check if input is a list with a single string element
    if (
        isinstance(input_value, list)
        and len(input_value) == 1
        and isinstance(input_value[0], str)
    ):
        # Extract the string from the list and process it
        input_value = input_value[0]

    if isinstance(input_value, str):
        # Remove any brackets or extra whitespace and split the string
        cleaned_input = (
            input_value.replace("[", "").replace("]", "").strip().replace("’", "").split()
        )
        try:
            # Convert the split string values to float
            formatted_value = [float(val) for val in cleaned_input]
        except ValueError:
            raise ValueError("String input for mean/std must be space-separated numbers.")
    elif isinstance(input_value, (list, np.ndarray)):
        try:
            # Convert elements to float
            formatted_value = [float(val) for val in input_value]
        except ValueError:
            raise ValueError("List or array input for mean/std must contain numbers.")
    else:
        raise TypeError("Input for mean/std must be a string, list, or numpy array.")

    if len(formatted_value) != 3:
        raise ValueError("Mean/std must have exactly three elements (for RGB channels).")

    return formatted_value
