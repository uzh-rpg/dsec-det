from pathlib import Path 
import cv2
from dsec_det.directory import DSECDirectory

from dsec_det.preprocessing import compute_img_idx_to_track_idx
from dsec_det.io import extract_from_h5
from dsec_det.visualize import render_object_detections_on_image, render_events_on_image


class DSECDet:
    def __init__(self, root: Path, split: str="train", sync: str="back", debug: bool=False,
                 load_detections_in_both_views: bool=False):
        """
        root: Root to the the DSEC dataset (the one that contains 'train' and 'test'
        split: Can be one of ['train', 'test']
        window_size: Number of microseconds of data
        sync: Can be either 'front' (last event ts), or 'back' (first event ts). Whether the front of the window or
              the back of the window is synced with the images.
        """
        assert root.exists()
        assert split in ['train', 'test']
        assert (root / split).exists()

        assert sync in ['front', 'back']

        self.debug = debug

        self.root = root / split
        self.sync = sync
        self.load_detections_in_both_views = load_detections_in_both_views

        self.height = 480
        self.width = 640

        self.directories = dict()
        self.img_idx_track_idxs = dict()

        self.subsequence_directories = sorted(list(self.root.glob("*/")))
        for f in self.subsequence_directories:
            directory = DSECDirectory(f)
            self.directories[f.name] = directory
            self.img_idx_track_idxs[f.name] = compute_img_idx_to_track_idx(directory.tracks.tracks['t'],
                                                                           directory.images.timestamps)

    def __len__(self):
        return sum(len(v)-1 for v in self.img_idx_track_idxs.values())

    def getitem(self, item, img_idx_to_track_idx, directory):
        output = {}

        # load image
        image_files = directory.images.image_files_distorted
        output['image'] = cv2.imread(str(image_files[item]))

        # find out where to load events
        if self.sync == "front":
            assert 0 < item < len(img_idx_to_track_idx)
            i_0 = item - 1
            i_1 = item
        else:
            assert 0 <= item < len(img_idx_to_track_idx)-1
            i_0 = item
            i_1 = item + 1

        # load events
        t_0, t_1 = directory.images.timestamps[[i_0, i_1]]
        output['events'] = extract_from_h5(directory.events.event_file, t_0, t_1)

        # load tracks
        tracks = directory.tracks.tracks
        if not self.load_detections_in_both_views:
            idx0, idx1 = img_idx_to_track_idx[item]
            output['tracks'] = tracks[idx0:idx1]
        else:
            idx0, idx1 = img_idx_to_track_idx[i_0]
            output['tracks_0'] = tracks[idx0:idx1]
            idx0, idx1 = img_idx_to_track_idx[i_1]
            output['tracks_1'] = tracks[idx0:idx1]

        if self.debug:
            # visualize tracks and events
            events = output['events']
            image = (255 * (output['image'].astype("float32") / 255) ** (1/2.2)).astype("uint8")
            output['debug'] = render_events_on_image(image, x=events['x'], y=events['y'], p=events['p'])
            output['debug'] = render_object_detections_on_image(output['debug'], output['tracks'])

        return output

    def __getitem__(self, item):
        for f in self.subsequence_directories:
            img_idx_to_track_idx = self.img_idx_track_idxs[f.name]
            if len(img_idx_to_track_idx)-1 <= item:
                item -= (len(img_idx_to_track_idx)-1)
                continue
            else:
                return self.getitem(item, img_idx_to_track_idx, self.directories[f.name])