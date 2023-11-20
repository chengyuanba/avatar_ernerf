import abc


class BaseMatting(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def image_matting(self):
        pass

    @abc.abstractmethod
    def video_matting(self):
        pass
