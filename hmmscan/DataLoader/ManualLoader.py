from typing import List, Dict
import numpy as np


class ManualLoader:
    """
    Create the data loader with a manual list as the sequence
    """
    def __init__(self, sequence: np.ndarray, directory: str, sequence_name: str, ae_type: str):
        self.sequences: List[np.ndarray] = [sequence.reshape(-1, 1)]
        self.metadata: Dict = {
            'dir': directory,
            'sequence_name': sequence_name,
            'ae_type': ae_type
        }
