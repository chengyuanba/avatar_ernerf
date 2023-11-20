import abc
import random
from typing import Union

import cv2
import numpy as np


class BaseParser(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def parsing(self,
                image: Union[str, np.ndarray]
                ) -> np.ndarray:
        """
        an abstract method need to be implemented.

        Params:
            image(str, numpy.ndarray): 图像文件路径 or 图像数据(BGR);

        Returns:
            label(numpy.ndarray): 分割得到的label图像, 值域[0, num_classes];

        """
        pass

    def draw_info(self,
                  image: np.ndarray,
                  label: np.ndarray,
                  color_list: list = None
                  ) -> np.ndarray:
        """
        Draw detection info.

        Params:
            image(numpy.ndarray): 图像数据, shape(H, W, 3), color(B, G, R), dtype"uint8";
            label(numpy.ndarray): 分割得到的label图像, shape(H, W);

        Returns:
            image_show(numpy.ndarray)

        """
        show = np.zeros_like(image, dtype="uint8")
        n_classes = int(np.max(label))

        if (color_list is None) or (len(color_list) < n_classes):
            color_list = [
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                for _ in range(n_classes+1)
            ]
        for cidx in range(1, n_classes+1):
            coords = np.where(label == cidx)
            if len(coords[0]) == 0:
                continue
            show[coords[0], coords[1], :] = color_list[cidx]

        show = cv2.addWeighted(image, 0.6, show, 0.4, 0)
        return show
