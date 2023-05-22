import yaml
import h5py
import numpy as np
import math
from pathlib import Path
import filecmp


# from https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
class dircmp(filecmp.dircmp):
    """
    Compare the content of dir1 and dir2. In contrast with filecmp.dircmp, this
    subclass compares the content of files with the same path.
    """
    def phase3(self):
        """
        Find out differences between common files.
        Ensure we are using content comparison with shallow=False.
        """
        fcomp = filecmp.cmpfiles(self.left, self.right, self.common_files,
                                 shallow=False)
        self.same_files, self.diff_files, self.funny_files = fcomp

def compare_dirs(dir1: Path, dir2: Path):
    """
    Compare two directory trees content.
    Return False if they differ, True is they are the same.
    """
    compared = dircmp(dir1, dir2)
    if (compared.left_only or compared.right_only or compared.diff_files
        or compared.funny_files):
        return False
    for subdir in compared.common_dirs:
        if not compare_dirs(dir1 / subdir, dir2 / subdir):
            return False
    return True


def extract_from_h5(h5file, t_min_us: int, t_max_us: int):
    with h5py.File(str(h5file), 'r') as h5f:
        ms2idx = np.asarray(h5f['ms_to_idx'], dtype='int64')
        t_offset = h5f['t_offset'][()]

        events = h5f['events']
        x = events['x']
        y = events['y']
        p = events['p']
        t = events['t']

        t_ev_start_us = t_min_us - t_offset
        assert t_ev_start_us >= t[0]
        t_ev_start_ms = math.floor(t_ev_start_us / 1000)
        ms2idx_start_idx = t_ev_start_ms
        ev_start_idx = ms2idx[ms2idx_start_idx]

        t_ev_end_us = t_max_us - t_offset
        assert t_ev_end_us <= t[-1]
        t_ev_end_ms = math.floor(t_ev_end_us / 1000)
        ms2idx_end_idx = t_ev_end_ms
        ev_end_idx = ms2idx[ms2idx_end_idx]

        x_new = x[ev_start_idx:ev_end_idx]
        y_new = y[ev_start_idx:ev_end_idx]
        p_new = p[ev_start_idx:ev_end_idx]
        t_new = t[ev_start_idx:ev_end_idx]

        t_ev_start_us_floored = t_ev_start_ms * 1000

        t_offset_new = t_offset + t_ev_start_us_floored
        t_new = t_new - t_ev_start_us_floored
        ms2idx_new = np.asarray(ms2idx[ms2idx_start_idx:ms2idx_end_idx + 1] - ms2idx[ms2idx_start_idx], dtype="uint64")

    # sanity checks
    assert ms2idx_new[math.floor((t_min_us - t_offset_new) / 1000)] == 0
    assert ms2idx_new[math.ceil((t_max_us - t_offset_new) / 1000)-1] == t_new.size

    output = {
        'p': p_new,
        't': t_new,
        'x': x_new,
        'y': y_new,
        't_offset': t_offset_new,
        'ms_to_idx': ms2idx_new,
    }
    return output

def h5_file_to_dict(h5_file: Path):
    with h5py.File(h5_file) as fh:
        return {k: fh[k][()] for k in fh.keys()}

def yaml_file_to_dict(yaml_file: Path):
    with yaml_file.open() as fh:
        return yaml.load(fh, Loader=yaml.UnsafeLoader)