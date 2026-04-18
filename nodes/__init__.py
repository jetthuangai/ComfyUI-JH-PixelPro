"""ComfyUI node wrappers — wrap core math into classes exposing INPUT_TYPES / RETURN_TYPES."""  # noqa: N999

from .detail_masker_node import JHPixelProHighFreqDetailMasker
from .facial_aligner_node import JHPixelProFacialAligner
from .frequency_node import JHPixelProFrequencySeparation
from .luminosity_node import JHPixelProLuminosityMasking
from .mask_refiner_node import JHPixelProSubPixelMaskRefiner
from .smoother_node import JHPixelProEdgeAwareSmoother

__all__ = [
    "JHPixelProEdgeAwareSmoother",
    "JHPixelProFacialAligner",
    "JHPixelProFrequencySeparation",
    "JHPixelProHighFreqDetailMasker",
    "JHPixelProLuminosityMasking",
    "JHPixelProSubPixelMaskRefiner",
]
