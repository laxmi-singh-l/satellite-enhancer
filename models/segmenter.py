# models/segmenter.py
import numpy as np

class LandCoverSegmenter:
    """
    Placeholder land-cover segmentation model.
    Currently returns an all-zero mask of the same HxW as the input image.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.int32)
