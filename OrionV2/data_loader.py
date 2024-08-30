# data_loader.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import read_with_pyav, select_random_frames_with_padding

class VideoDataset(Dataset):
    def __init__(self, config, split):
        self.model_name = config.model_name
        self.label_column = config.label_column
        self.frames_size = config.frames_size
        self.num_frames = config.num_frames
        self.split = split
        
        df = pd.read_csv(config.dataset_path, sep=config.separator, engine="python")

        # Filter based on split
        df = df[df[config.split_column] == self.split]

        # Filter based on FPS
        if config.fps != 0:
            df = df[df[config.fps_column] == config.fps]
        
        self.inMemory = config.in_memory

        # Determine dataset bounds based on `dataset_origin`
        if config.dataset_origin == "UCSF":
            df_filtered = df[df['UCSF_StudyID'].notna()]
            lowerBound, upperBound = 0, len(df_filtered)

        elif config.dataset_origin == "MHI":
            df_filtered = df[df['UCSF_StudyID'].isna()]
            lowerBound, upperBound = 0, len(df_filtered)

        elif config.dataset_origin == "BOTH":
            df_filtered = df
            lowerBound, upperBound = 0, len(df_filtered)

        elif config.dataset_origin == "UCSF_CLIPPED":
            df_filtered = df[df['UCSF_StudyID'].notna()]
            df_filtered = self._apply_clipping_logic(df_filtered)
            lowerBound, upperBound = 0, len(df_filtered)

        elif config.dataset_origin == "MHI_CLIPPED":
            df_filtered = df[df['UCSF_StudyID'].isna()]
            df_filtered = self._apply_clipping_logic(df_filtered)
            lowerBound, upperBound = 0, len(df_filtered)

        elif config.dataset_origin == "MHI_CLIPPED_NO_MODULO":
            df_filtered = df[df['UCSF_StudyID'].isna()]
            df_filtered = self._apply_clipping_logic(df_filtered, modulo_5=config.modulo_5)
            lowerBound, upperBound = 0, len(df_filtered)

        elif config.dataset_origin == "MHI_SUB_40":
            df_filtered = df[df['UCSF_StudyID'].isna()]
            df_filtered = df[df[self.label_column] < 40]
            # save plot of the distribution of the labels
            
            lowerBound, upperBound = 0, len(df_filtered)

        else:
            raise ValueError(f"Unknown dataset_origin: {config.dataset_origin}")

        # Handle `howMany` parameter
        if config.how_many > 0:
            upperBound = config.how_many

        if self.inMemory:
            # Load all videos into memory
            self.videos = [self._get_video(video_path) for video_path in tqdm(df_filtered[config.filename_column][lowerBound:upperBound], desc="Loading videos into memory")]
        else:
            # Just record the file paths (no in-memory loading)
            self.videos = list(df_filtered[config.filename_column][lowerBound:upperBound])
        self.labels = list(df_filtered[config.label_column][lowerBound:upperBound] / (100 if (config.problem_type  == "regression" and not config.label_normalized) else 1))

        
        ########################################################
        # SPECIFIC TO TEST SET
        ########################################################
        if self.split == "TEST":
            # Load exams IDs to be able to compute metrics on the exam level
            if "MHI" in config.dataset_origin:
                self.exams = list(df_filtered['StudyInstanceUID'][lowerBound:upperBound])
            elif "UCSF" in config.dataset_origin:
                self.exams = list(df_filtered['UCSF_StudyID'][lowerBound:upperBound])
            else:
                self.exams = list(df_filtered['UCSF_StudyID'][lowerBound:df_filtered['UCSF_StudyID'].notna().sum()])
                self.exams.extend(df_filtered['StudyInstanceUID'][df_filtered['UCSF_StudyID'].notna().sum():])
            
            # Load the number of frame to be able to run the model on the most subset of each video
            tmp_labels = []
            tmp_videos = []
            tmp_exams = []
            self.starting_index = []
            for i, (_, row) in enumerate(df_filtered[lowerBound:upperBound].iterrows()):
                if row.NumberOfFrames > config.num_frames:
                    for j in range(0,int(row.NumberOfFrames-config.num_frames)+1, config.stride):
                        tmp_videos.append(self.videos[i])
                        tmp_labels.append(self.labels[i])
                        tmp_exams.append(self.exams[i])
                        self.starting_index.append(j)
                else:
                    tmp_videos.append(self.videos[i])
                    tmp_labels.append(self.labels[i])
                    tmp_exams.append(self.exams[i])
                    self.starting_index.append(0)
            self.videos = tmp_videos
            self.labels = tmp_labels
            self.exams = tmp_exams
        

    def _apply_clipping_logic(self, df, modulo_5=True):
        """
        Apply the clipping logic for MHI_CLIPPED and UCSF_CLIPPED.
        Clipping is performed here based on the condition that config.label_column modulo 5 equals 0.
        """
        pd.set_option('mode.chained_assignment', None) 
        df[self.label_column] = df[self.label_column].astype(int)
        if modulo_5:
            df_modulated = df[df[self.label_column] % 5 == 0]
        else:
            df_modulated = df

        seed_value = 42
        filtered_samples = []
        # Define value ranges or specific values for clipping
        value_classes_to_filter = [50, 55, 60]
        for value in value_classes_to_filter:
            value_group = df_modulated[df_modulated[self.label_column] == value]
            if len(value_group) > 2000:
                sampled_group = value_group.sample(n=2000, random_state=seed_value)
            else:
                sampled_group = value_group
            filtered_samples.append(sampled_group)

        remaining_df = df_modulated[~df_modulated[self.label_column].isin(value_classes_to_filter)]
        df = pd.concat([remaining_df] + filtered_samples)

        return df

    def getVideos(self, idx):
        video = read_with_pyav(self.videos[idx], self.model_name, self.frames_size)
        if self.split == "TEST":
            video = video[self.starting_index[idx]:self.starting_index[idx]+self.num_frames, :, :, :]
            # Pad the video if it has less frames than the required number of frames
            if video.shape != self.num_frames: 
                padding = np.zeros((self.num_frames - video.shape[0], *video.shape[1:]), dtype=video.dtype)
                video = np.concatenate((video, padding), axis=0)
        else:
            video = select_random_frames_with_padding(video, num_frames=self.num_frames)
        return video.transpose(3, 0, 1, 2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.inMemory:
            video = self.videos[idx]
        else:
            video = self.getVideos(idx)
        label = self.labels[idx]
        if self.split == "TEST":
            if type(self.exams[idx]) != str:
                exam_id = str(int(self.exams[idx]))
            else:
                exam_id = self.exams[idx]
            return video, label, exam_id, self.videos[idx]
        else:
            return video, label


class DumbVideoDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        randomValue = np.random.rand()
        return np.ones((3, 32, 182, 182)) * randomValue, randomValue