# models/super_resolution.py
import numpy as np

class IRSuperResolution:
    """
    Placeholder super-resolution model for IR images.
    Currently returns the input unchanged.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, ir_image: np.ndarray) -> np.ndarray:
        # TODO: replace with real super-resolution logic
        return ir_image
