import abc
from typing import Union
import random

import cv2
import numpy as np


class BaseDetector(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def detection(self,
                  image: Union[str, np.ndarray]
                  ) -> any:
        """
        an abstract method need to be implemented.

        Params:
            image(str, numpy.ndarray): 图像文件路径 or 图像数据(BGR);

        Returns:
            any
        """
        pass

    def draw_info(self,
                  image: np.ndarray,
                  bboxes: np.ndarray,
                  labels: list,
                  scores: np.ndarray,
                  bbox_colors: dict = {},
                  text_color: tuple = (225, 255, 255),
                  ) -> np.ndarray:
        """
        Draw detection info.

        Params:
            image(numpy.ndarray): 图像数据, shape(H, W, 3), color(B, G, R), dtype"uint8";
            bboxes(numpy.ndarray): 检测得到的目标边框, shape(n, 4), [x1, y1, x2, y2];
            labels(list): 每个边框对应的类别名, list[str, str, ...];
            scores(numpy.ndarray): 每个边框对应的置信度, shape(n, 1);

        Returns:
            image_show(numpy.ndarray)

        """
        if len(bbox_colors) == 0:
            for label_name in labels:
                if label_name in bbox_colors:
                    continue
                else:
                    bbox_colors[label_name] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )

        image_h, image_w = image.shape[:2]
        th01 = round(0.002 * (image_h + image_w) / 2) + 1
        th02 = max(th01 - 1, 1)
        for bbox, label, score in zip(bboxes, labels, scores):
            # draw bbox
            c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(image, c1, c2, color=bbox_colors[label],
                          thickness=th01, lineType=cv2.LINE_AA)
            # draw score
            text = "{}:{:.4f}".format(label, float(score[0]))
            text_size = cv2.getTextSize(text, 0, fontScale=th01/3,
                                        thickness=th02)[0]
            c2 = (c1[0]+text_size[0], c1[1]-text_size[1]-3)
            cv2.rectangle(image, c1, c2, color=bbox_colors[label],
                          thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(image, text, (c1[0], c1[1]-2), 0,
                        th01/3, color=text_color,
                        thickness=th02, lineType=cv2.LINE_AA)
        return image
