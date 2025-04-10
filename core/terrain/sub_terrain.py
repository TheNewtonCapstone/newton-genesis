import numpy as np


class SubTerrain:
    def __init__(
        self,
        width: int = 256,
        length: int = 256,
        vertical_scale: float = 1.0,
        horizontal_scale: float = 1.0,
    ):
        self.width: int = width
        self.length: int = length

        self.vertical_scale: float = vertical_scale
        self.horizontal_scale: float = horizontal_scale

        self.height_field: np.ndarray = np.zeros(
            (self.width, self.length),
            dtype=np.int16,
        )