# models/__init__.py
"""
Convenience imports for the models package.

Only import symbols that are actually defined in the submodules.
Optional modules are imported lazily via try/except so the package
still works even if some files are missing during development.
"""

from .detector import LANDSAT_CLASSES

# Enhancement-related utilities (defined in models/enhance.py)
from .enhance import (
    enhance_image,
    enhance_image_with_model,
    load_enhancement_model,
    load_enhancement_model_from_path,
)

# IR -> RGB colorizer (defined in models/colorize.py)
from .colorize import IR2RGB

# Optional: land-cover segmenter
try:
    from .segmenter import LandCoverSegmenter
except ImportError:  # optional
    LandCoverSegmenter = None

# Optional: IR super-resolution model
try:
    from .super_resolution import IRSuperResolution
except ImportError:  # optional
    IRSuperResolution = None
