"""orion-Dynamic Dataset."""
import collections
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytorchvideo.transforms
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
                ``Filename'' (string): filename of video
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
        split="train",
        target_label="HCM",
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
        if not isinstance(target_label, list):
            target_label = [target_label]
        self.target_label = target_label
        self.mean = mean
        self.std = std
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
            os.path.join(self.folder, self.filename), sep="µ", engine="python"
        )
        self.header = list(df_dataset.columns)  # Add this line to store the DataFrame's header
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
            if split in ["all", fileMode] and os.path.exists(fileName):
                self.fnames.append(fileName)
                self.outcome.append(row)

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

        if self.split == "inference":
            target = None
        else:
            # Gather targets
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
                    if self.split == "clinical_test" or self.split == "external_test":
                        # change this
                        #     target.append(np.float32(0))
                        # else:
                        #     target.append(np.float32(self.outcome[index][self.header.index(t)]))
                        # change this
                        target.append(self.outcome[index].iloc[self.header.index(t)])
                    else:
                        target.append(self.outcome[index].iloc[self.header.index(t)])

                target = [np.float32(i) for i in target]
        if target is not None:
            if target != []:
                target = tuple(target) if len(target) > 1 else target[0]
                if self.target_transform is not None:
                    target = self.target_transform(target)

        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

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

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class Echo_Multi(torch.utils.data.Dataset):
    """orion Dataset.
    Args:
        root (string): Root directory of dataset (defaults to `orion.config.DATA_DIR`)
        split (string): One of {"train", "val", "test", "external_test"}
        target_label (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
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
        root="../../data/",
        data_filename=None,
        split="train",
        target_label="HCM",
        datapoint_loc_label="FileName",
        resize=224,
        view_count="view_count",
        mean=0.0,
        std=1.0,
        length=32,
        period=1,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        video_transforms=None,
        num_classes=None,
        target_transform=None,
        external_test_location=None,
    ) -> None:
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split
        # if not isinstance(target_label, list):
        #     target_label = [target_label]
        self.target_label = target_label
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.video_transforms = video_transforms
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.resize = resize
        self.view_count = view_count

        self.fnames, self.outcome = [], []
        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            with open(os.path.join(self.folder, self.filename)) as f:
                self.header = f.readline().strip().split("µ")
                view_countIndex = self.header.index(self.view_count)
                # target_index = self.header.index(self.target_label)
                # filenameIndex = self.header.index(self.datapoint_loc_label + str(0))
                splitIndex = self.header.index("Split")
                target_index = self.header.index(target_label)
                filenameIndex = []
                outcomeIndex = []
                for line in f:
                    lineSplit = line.strip().split("µ")
                    view_count = int(lineSplit[view_countIndex])
                    for i in range(view_count):
                        filenameIndex.append(self.header.index(self.datapoint_loc_label + str(i)))
                        outcomeIndex.append(self.header.index(self.target_label))
                    fileMode = lineSplit[splitIndex].lower()
                    fileName = []
                    outcomes = []
                    for i in range(view_count):
                        fileName.append(lineSplit[filenameIndex[i]])
                        outcomes.append(lineSplit[outcomeIndex[i]])
                    file_vids = []
                    for i in fileName:
                        if split in ["all", fileMode] and os.path.exists(i):
                            file_vids.append(i)
                    if len(file_vids) > 1:
                        self.fnames.append(file_vids)
                        self.outcome.append(outcomes)
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

    #         # define weights for weighted sampling
    #         labels = np.array([self.outcome[ind][target_index] for ind in range(len(self.outcome))], dtype=int)
    #         weights = 1 - (np.bincount(labels) / len(labels))

    #         self.weight_list = np.zeros(len(labels))

    #         for label in range(len(weights)):
    #             weight = weights[label]
    #             self.weight_list[np.where(labels == label)] = weight

    def make_video(self, video, count):
        # Find filename of video
        # if self.split == "external_test":
        #     video = os.path.join(self.external_test_location, self.fnames[index])
        # elif self.split == "clinical_test":
        #     video = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        # else:
        #     video = self.fnames[index]

        # Load video into np.array
        # video = orion.utils.loadvideo(video).astype(np.float32)
        # transforms
        video = orion.utils.loadvideo(video).astype(np.float32)

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
        #       apply normalization before augmentaiton per riselab and pytorchvideo
        #       first initialization is a float of 0 so need to avoid indexing a float
        if isinstance(self.mean, float):
            video = v2.Normalize(self.mean, self.std)(video)
        else:
            video = v2.Normalize(self.mean[count], self.std[count])(video)

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
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

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
        return video

    # END HERE SEAN
    def make_targets(self, index):
        # Gather targets
        target = []
        for t in self.target_label:
            if t == "Filename1":
                target.append(self.fnames[index][1])
            else:
                if self.split == "clinical_test" or self.split == "external_test":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index]))

            if target != []:
                target = tuple(target) if len(target) > 1 else target[0]
                if self.target_transform is not None:
                    target = self.target_transform(target)

            return target

    def __getitem__(self, index):
        if self.split == "external_test":
            for i in self.fnames[index][0]:
                video = os.path.join(self.external_test_location, i)
        elif self.split == "clinical_test":
            for i in self.fnames[index][0]:
                video = os.path.join(self.folder, "ProcessedStrainStudyA4c", i)
        else:
            vids_stack = []
            targets = []
            filenames = []
            if self.split == "inference":
                for count, i in enumerate(self.fnames[index]):
                    video = Echo_Multi.make_video(self, i, count)
                    vids_stack.append(video)
                    filenames.append(i)
                return vids_stack, filenames
            else:
                for count, i in enumerate(self.fnames[index]):
                    # print(len(self.fnames[index]))
                    # print(self.fnames[index])
                    video = Echo_Multi.make_video(self, i, count)
                    vids_stack.append(video)
                    target = Echo_Multi.make_targets(self, index)
                    targets.append(target)
                    filenames.append(i)
                # return vids_stack, targets[1][1], filenames
                return vids_stack, targets[1][1]

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
