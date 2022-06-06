from abc import ABC, abstractmethod

import jittor as jt

from .functional import gamma, identity, nearest_pixel


class _OOPWrapper(jt.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    @abstractmethod
    def execute(self, coords):
        pass


class Gamma(_OOPWrapper):
    """See ``torchsample.encoding.functional.gamma``."""

    def execute(self, coords):
        return gamma(coords, **self.kwargs)


class Identity(_OOPWrapper):
    """See ``torchsample.encoding.functional.gamma``."""

    def execute(self, coords):
        return identity(coords, **self.kwargs)


class NearestPixel(_OOPWrapper):
    """See ``torchsample.encoding.functional.nearest_pixel``."""

    def execute(self, coords):
        return nearest_pixel(coords, **self.kwargs)
