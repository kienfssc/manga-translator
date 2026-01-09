from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class BaseSegmentor(ABC):
    """
    Abstract Base Class for all detection tasks (Text, Bubbles, etc.)
    """

    @abstractmethod
    def load_model(self, model_path: str):
        """Load model weights into memory/GPU."""
        pass

    @abstractmethod
    def segment(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on an image.
        Returns: List of dicts: [{'box': [x1, y1, x2, y2], 'label': 'text', 'conf': 0.9}]
        """
        pass
