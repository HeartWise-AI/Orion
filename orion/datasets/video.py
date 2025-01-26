import collections
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from torchvision.transforms import v2

# Global variable for directory paths
dir2 = os.path.abspath("/volume/Orion/orion")
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)

import orion


class Video(torch.utils.data.Dataset):
    """
    A dataset class for handling orion video data.

    Args:
        root (str): Root directory of the dataset. Defaults to '../../data/'.
        data_filename (str): Name of the data file. Defaults to None.
        split (str): Dataset split to use ('train', 'val', 'test'). Defaults to 'train'.
        target_label (str): Name of the target label column. Defaults to None.
        datapoint_loc_label (str): Column name for file paths. Defaults to 'FileName'.
        -ize (int): Size to resize videos to (height, width). Defaults to 224.
        mean (float or np.array): Channel-wise means for normalization. Defaults to 0.0.
        std (float or np.array): Channel-wise stds for normalization. Defaults to 1.0.
        length (int): Number of frames to clip from the video. Defaults to 32.
        period (int): Sampling period for frame clipping. Defaults to 1.
        max_length (int): Max number of frames to clip. Defaults to 250.
        pad (int): Number of pixels for padding frames. Defaults to None.
        noise (float): Fraction of pixels to black out as noise. Defaults to None.
        video_transforms (list): List of torchvision transforms. Defaults to None.
        rand_augment (bool): Apply random augmentation. Defaults to False.
        apply_mask (bool): Apply masking to videos. Defaults to False.
        target_transform (callable): Transform to apply to the target. Defaults to None.
        external_test_location (str): Path for external testing videos. Defaults to None.

    Returns:
        Tuple: normalized, resized, and augmented videos with labels.

    Note:
        Augmentations, normalizations, frame sampling, and resizing have been
        rigorously tested and visualized.
    """

    def __init__(
        self,
        root="../../data/",
        data_filename=None,
        split="train",
        target_label=None,
        datapoint_loc_label="FileName",
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
        rand_augment=False,
        apply_mask=False,
        target_transform=None,
        external_test_location=None,
        weighted_sampling=False,
        normalize=True,
        debug=False,
    ) -> None:
        # Initialize instance variables
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split

        # Handle target labels for multi-head case
        if isinstance(target_label, str):
            self.target_label = [target_label]
        elif isinstance(target_label, dict):
            # If target_label is a dictionary (head_structure), extract the keys
            self.target_label = list(target_label.keys())
        elif isinstance(target_label, list):
            self.target_label = target_label
        else:
            self.target_label = None

        self.mean = format_mean_std(mean)
        self.std = format_mean_std(std)
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.video_transforms = video_transforms
        self.rand_augment = rand_augment
        self.apply_mask = apply_mask
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.resize = resize
        self.weighted_sampling = weighted_sampling
        self.debug = debug
        self.normalize = normalize

        self.fnames, self.outcome = [], []
        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            with open(os.path.join(self.folder, self.filename)) as f:
                self.header = f.readline().strip().split("α")
                if len(self.header) == 1:
                    raise ValueError(
                        "Header was not split properly. Please ensure the file uses 'α' (alpha) as the delimiter."
                    )

            self.fnames, self.outcomes, self.target_indices = self.load_data(
                split, self.target_label
            )
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

        if self.weighted_sampling is True:
            # For multi-head, we'll use the first target for weighted sampling
            if isinstance(self.outcome[0], dict):
                first_target = list(self.outcome[0].keys())[0]
                labels = np.array([outcome[first_target] for outcome in self.outcome], dtype=int)
            else:
                labels = np.array([outcome for outcome in self.outcome], dtype=int)

            weights = 1 - (np.bincount(labels) / len(labels))
            self.weight_list = np.zeros(len(labels))

            for label in range(len(weights)):
                weight = weights[label]
                self.weight_list[np.where(labels == label)] = weight

    def load_data(self, split, target_labels):
        """
        Load data from the CSV file and extract filenames and outcomes.

        Args:
            split (str): Dataset split ('train', 'val', 'test', 'all')
            target_labels (list): List of target label column names

        Returns:
            tuple: (filenames, outcomes, target_indices)
        """
        # Read the "α" separated file using pandas
        file_path = os.path.join(self.folder, self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")

        # Handle target indices for multi-head case
        if target_labels is None:
            target_indices = None
        else:
            target_indices = {}
            for label in target_labels:
                try:
                    target_indices[label] = data.columns.get_loc(label)
                except KeyError:
                    print(f"Warning: Target label '{label}' not found in data columns")
                    continue

        self.fnames = []
        self.outcome = []

        # Iterate through rows using iterrows
        for index, row in data.iterrows():
            file_name = row.iloc[filename_index]
            file_mode = row.iloc[split_index].lower()

            if split in ["all", file_mode] and os.path.exists(file_name):
                self.fnames.append(file_name)
                if target_indices is not None:
                    # For multi-head, create a dictionary of outcomes
                    outcomes = {}
                    for label, idx in target_indices.items():
                        outcomes[label] = row.iloc[idx]
                    self.outcome.append(outcomes)

        return self.fnames, self.outcome, target_indices

    def __getitem__(self, index):
        # Find filename of video
        video_fname = self.fnames[index]

        video = orion.utils.loadvideo(video_fname).astype(np.float32)

        if self.apply_mask:
            path = video_fname.rsplit("/", 2)
            mask_filename = f"{path[0]}/mask/{path[2]}"
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
            video = v2.Resize((self.resize, self.resize), antialias=True)(video)

        if self.normalize == True:
            if hasattr(self, "mean") and hasattr(self, "std"):
                video = v2.Normalize(self.mean, self.std)(video)

        if self.video_transforms is not None:
            transforms = v2.RandomApply(torch.nn.ModuleList(self.video_transforms), p=0.5)
            scripted_transforms = torch.jit.script(transforms)
            try:
                video = scripted_transforms(video)
            except RuntimeError as e:
                print(f"Skipping video {self.fnames[index]} due to error: {str(e)}")
                return self.__getitem__(index + 1)  # retry with next sample

        if self.rand_augment:
            raug = [
                v2.RandAugment(
                    magnitude=9,
                    num_layers=2,
                    prob=0.5,
                    sampling_type="gaussian",
                    sampling_hparas=None,
                )
            ]
            raug_composed = v2.Compose(raug)
            video = raug_composed(video)

        # Permute the tensor to have the shape [F, H, W, C]
        video = video.permute(1, 0, 2, 3)
        video = video.numpy()

        # Set number of frames
        c, f, h, w = video.shape
        length = f // self.period if self.length is None else self.length

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

        if self.target_label is not None:
            # Handle multi-head case
            if isinstance(self.outcome[index], dict):
                # For multi-head, return the dictionary of targets
                target = self.outcome[index]
                if self.target_transform is not None:
                    target = {
                        k: self.target_transform(torch.tensor(v).float())
                        for k, v in target.items()
                    }
            else:
                # Original single-head logic with all special cases
                target = []
                for t in self.target_label:
                    key = os.path.splitext(self.fnames[index])[0]
                    if t == "Filename":
                        target.append(self.fnames[index])
                    elif t == "LargeIndex":
                        # Traces are sorted by cross-sectional area
                        # Largest (diastolic) frame is last
                        target.append(np.int(self.frames[key][-1]))
                    elif t == "SmallIndex":
                        # Largest (diastolic) frame is first
                        target.append(np.int(self.frames[key][0]))
                    elif t == "LargeFrame":
                        target.append(video[:, self.frames[key][-1], :, :])
                    elif t == "SmallFrame":
                        target.append(video[:, self.frames[key][0], :, :])
                    elif t in ["LargeTrace", "SmallTrace"]:
                        if t == "LargeTrace":
                            t = self.trace[key][self.frames[key][-1]]
                        else:
                            t = self.trace[key][self.frames[key][0]]
                        x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                        x = np.concatenate((x1[1:], np.flip(x2[1:])))
                        y = np.concatenate((y1[1:], np.flip(y2[1:])))

                        r, c = skimage.draw.polygon(
                            np.rint(y).astype(np.int),
                            np.rint(x).astype(np.int),
                            (video.shape[2], video.shape[3]),
                        )
                        mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                        mask[r, c] = 1
                        target.append(mask)
                    else:
                        target.append(self.outcome[index])

                target = [np.float32(i) for i in target]
                if target != []:
                    target = tuple(target) if len(target) > 1 else target[0]
                    if self.target_transform is not None:
                        target = self.target_transform(target)

        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        video = video[0] if self.clips == 1 else np.stack(video)
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

        if self.split == "inference":
            return video, self.fnames[index]

        return video, target, video_fname

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Video dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


import collections
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from torchvision.transforms import v2

# Global variable for directory paths
dir2 = os.path.abspath("/volume/Orion/orion")
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)

import orion


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    Used to avoid issues with Windows if anonymous."""
    return collections.defaultdict(list)


def format_mean_std(input_value):
    """
    Formats the mean or std value to a list of floats with length=3.
    """
    if input_value is 1.0 or input_value is 0.0:
        print("Mean/STD value is not defined or is trivial.")
        return input_value

    # If it's a single-element list containing a string, unwrap it.
    if (
        isinstance(input_value, list)
        and len(input_value) == 1
        and isinstance(input_value[0], str)
    ):
        input_value = input_value[0]

    if isinstance(input_value, str):
        cleaned_input = (
            input_value.replace("[", "").replace("]", "").strip().replace("’", "").split()
        )
        try:
            formatted_value = [float(val) for val in cleaned_input]
        except ValueError:
            raise ValueError("String input for mean/std must be space-separated numbers.")
    elif isinstance(input_value, (list, np.ndarray)):
        try:
            formatted_value = [float(val) for val in input_value]
        except ValueError:
            raise ValueError("List or array input for mean/std must contain numbers.")
    else:
        raise TypeError("Input for mean/std must be a string, list, or numpy array.")

    if len(formatted_value) != 3:
        raise ValueError("Mean/std must have exactly three elements (for RGB channels).")


class Video_Multi(torch.utils.data.Dataset):
    """
    A multi-view video dataset class that supports single or multi targets.
    It can convert string labels (like "Normal") to numeric if 'labels_map'
    is provided in the config.

    Returns:
      (list_of_view_tensors, dict_of_label_tensors, file_list).
    """

    def __init__(
        self,
        root="../../data/",
        data_filename=None,
        split="train",
        target_label=None,  # e.g. "y_true_cat_label" or a dict
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
        labels_map=None,
    ):
        super().__init__()
        self.root = pathlib.Path(root)
        self.filename = data_filename
        self.split = split
        self.datapoint_loc_label = datapoint_loc_label
        self.view_count = view_count
        self.resize = resize
        self.mean = format_mean_std(mean)
        self.std = format_mean_std(std)
        self.length = length
        self.period = period
        self.max_length = max_length
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.video_transforms = video_transforms
        self.apply_mask = apply_mask
        self.rand_augment = rand_augment
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.weighted_sampling = weighted_sampling
        self.debug = debug
        self.normalize = normalize

        # We'll store labels_map so we can look up any string -> numeric
        # For example:
        # labels_map = {
        #    "y_true_cat_label": {
        #        "Normal": 0,
        #        "Physiology abnormal": 1
        #    }
        # }
        self.labels_map = labels_map or {}

        # --- Parse target_label -> self.target_label_list
        if isinstance(target_label, str):
            self.target_label_list = [target_label]
        elif isinstance(target_label, dict):
            self.target_label_list = list(target_label.keys())
        elif isinstance(target_label, list):
            self.target_label_list = target_label
        else:
            self.target_label_list = []

        # Read CSV with "α" delimiter
        df_dataset = pd.read_csv(self.root / self.filename, sep="α", engine="python")

        # Build a dictionary of column indices for each label
        self.target_indices = {}
        for lbl in self.target_label_list:
            if lbl in df_dataset.columns:
                self.target_indices[lbl] = df_dataset.columns.get_loc(lbl)
            else:
                print(f"Warning: label '{lbl}' not found in CSV columns. Skipping it.")

        # Identify 'Split' column
        if "Split" not in df_dataset.columns:
            raise ValueError("CSV must have a 'Split' column.")
        splitIndex = df_dataset.columns.get_loc("Split")

        # Identify columns for FileName0, FileName1, etc.
        filename_indices = []
        for i in range(self.view_count):
            colname = f"{self.datapoint_loc_label}{i}"
            if colname in df_dataset.columns:
                filename_indices.append(df_dataset.columns.get_loc(colname))
            else:
                filename_indices.append(None)
                print(f"Warning: Column '{colname}' not found in CSV. This view is skipped.")

        self.fnames = []
        self.outcome = []

        # Filter rows by split
        for _, row in df_dataset.iterrows():
            file_mode = str(row.iloc[splitIndex]).lower()
            if split not in ["all", file_mode]:
                continue

            # Gather the paths for each view
            view_files = []
            # Get filenames for each view
            file_names = []
            for fi in filename_indices:
                if fi is not None:
                    file_name = row.iloc[fi]
                    if isinstance(file_name, str) and os.path.exists(file_name):
                        file_names.append(file_name)
                    else:
                        file_names.append(None)
                else:
                    file_names.append(None)

            # Only append if we have valid files for at least 2 views
            valid_files = [f for f in file_names if f is not None]
            if len(valid_files) > 1:
                self.fnames.append(file_names)

                # Build outcome dict
                outcome_dict = {}
                for lbl, idx in self.target_indices.items():
                    raw_val = row.iloc[idx]
                    outcome_dict[lbl] = raw_val
                self.outcome.append(outcome_dict)

        # Weighted sampling logic if needed ...
        # e.g. self.weight_list = ...
        if self.debug:
            print(f"[Video_Multi] Found {len(self.fnames)} samples for split={split}")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Get filenames for all views of the current sample
        video_fnames = self.fnames[index]

        # List to hold processed video tensors for each view
        view_tensors = []

        for fname in video_fnames:
            # Load the video for this view
            video_np = orion.utils.loadvideo(fname).astype(np.float32)

            # Apply mask if enabled
            if self.apply_mask:
                path = fname.rsplit("/", 2)
                mask_filename = f"{path[0]}/mask/{path[2]}"
                mask_filename = mask_filename.split(".avi")[0] + ".npy"
                mask = np.load(mask_filename).transpose(2, 0, 1)

                # Apply mask to the video (code from original Video class)
                length = video_np.shape[1]
                if mask.shape[1] < length:
                    mask = np.pad(mask, [(0, 0), (length - mask.shape[1], 0), (0, 0)])
                if mask.shape[2] < length:
                    mask = np.pad(mask, [(0, 0), (0, 0), (length - mask.shape[2], 0)])
                mask = mask[:, :length, :length]

                for ind in range(video_np.shape[0]):
                    video_np[ind, :, :, :] *= mask

            # Add noise if specified
            if self.noise is not None and self.noise > 0:
                n = video_np.shape[1] * video_np.shape[2] * video_np.shape[3]
                ind = np.random.choice(n, round(self.noise * n), replace=False)
                f = ind % video_np.shape[1]
                ind //= video_np.shape[1]
                i = ind % video_np.shape[2]
                j = ind // video_np.shape[2]
                video_np[:, f, i, j] = 0

            # Convert to tensor and resize
            video = torch.from_numpy(video_np)
            if self.resize is not None:
                video = v2.Resize((self.resize, self.resize), antialias=True)(video)

            # Normalize
            if self.normalize and self.mean is not None and self.std is not None:
                video = v2.Normalize(self.mean, self.std)(video)

            # Apply video transforms
            if self.video_transforms:
                transform = v2.RandomApply(torch.nn.ModuleList(self.video_transforms), p=0.5)
                scripted = torch.jit.script(transform)
                video = scripted(video)

            # Apply RandAugment
            if self.rand_augment:
                raug = v2.RandAugment(magnitude=9, num_ops=2)
                video = raug(video)

            # Handle frame count and period
            F, C, H, W = video.shape  # Original frame count

            # Calculate total frames needed BEFORE downsampling
            frames_needed = self.length * self.period

            if F < frames_needed:
                # Pad with zeros
                pad_size = frames_needed - F
                video = torch.cat(
                    [video, torch.zeros((pad_size, C, H, W), dtype=video.dtype)], dim=0
                )
            elif F > frames_needed:
                # Random temporal crop
                start = np.random.randint(0, F - frames_needed)
                video = video[start : start + frames_needed]

            # Now downsample by period
            if self.period > 1:
                video = video[:: self.period]  # This should give exactly self.length frames

            # Final validation
            assert video.shape[0] == self.length, (
                f"Final frame count mismatch: {video.shape[0]} vs {self.length}. "
                f"Check length={self.length} and period={self.period} configuration."
            )

            view_tensors.append(video)

        # Get target labels
        target = {}
        for lbl in self.target_label_list:
            raw_val = self.outcome[index].get(lbl, None)
            # Convert string labels to numeric using labels_map if provided
            if lbl in self.labels_map and isinstance(raw_val, str):
                target[lbl] = self.labels_map[lbl].get(raw_val, raw_val)
            else:
                target[lbl] = raw_val

        return view_tensors, target, video_fnames

    def plot_middle_frame(self, x, title, index):
        # Only plot for every 10th example
        if index % 10 == 0:
            # Calculate the index of the middle frame
            print(f"Min: {x.min()}, Max: {x.max()}, Title: {title}")

            middle_frame_index = x.shape[0] // 2

            # Select the middle frame
            frame = x[middle_frame_index, :, :, :]

            # Convert the numpy array to a PyTorch tensor if it's not already a tensor
            if not isinstance(frame, torch.Tensor):
                frame = torch.from_numpy(frame)

            # Permute the tensor dimensions
            frame = frame.permute(1, 2, 0)

            # Convert the tensor to a numpy array
            frame = frame.cpu().numpy()

            # Normalize the pixel values to the range [0, 1] for plotting
            frame = (frame - frame.min()) / (frame.max() - frame.min())

            # Plot the frame
            plt.imshow(frame)
            plt.title(title)
            plt.axis("off")
            plt.show()
