"""Core math — pure tensor in, pure tensor out. Không PIL/cv2/numpy ở hot path."""

from .frequency import frequency_separation

__all__ = ["frequency_separation"]
