import numpy as np


class BaseDirectory:
    def __init__(self, root):
        self.root = root


class DSECDirectory:
    def __init__(self, root):
        self.images = ImageDirectory(root / "images")
        self.events = EventDirectory(root / "events")
        self.tracks = TracksDirectory(root / "object_detections")


class ImageDirectory(BaseDirectory):
    @property
    def timestamps(self):
        return np.genfromtxt(self.root / "timestamps.txt")

    @property
    def image_files_rectified(self):
        return sorted(list((self.root / "left/rectified").glob("*.png")))

    @property
    def image_files_distorted(self):
        return sorted(list((self.root / "left/distorted").glob("*.png")))


class EventDirectory(BaseDirectory):
    @property
    def event_file(self):
        return self.root / "left/events.h5"


class TracksDirectory(BaseDirectory):
    @property
    def tracks(self):
        return np.load(self.root / "left/tracks.npy")