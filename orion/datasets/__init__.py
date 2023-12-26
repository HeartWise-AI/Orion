"""
Defines datasets for loading video models
"""

from orion.datasets.video import Video, Video_Multi
from orion.datasets.video_distribution import Video_dist
from orion.datasets.video_feature import Video_feature

__all__ = ["Video", "Video_Multi", "Video_feature"]
