import numpy as np

class Image:
    def __init__(self, img: np.array, filename: str) -> None:
        self._img = img
        self._filename = filename

    @property
    def img(self)-> np.array:
        return self._img

    @property
    def filename(self)-> str:
        return self._filename