import abc
from typing import Tuple, Union

import cv2
import numpy as np

COCO_BODY_17 = {
    "body": (
        (15, 13), (13, 11), (16, 14), (14, 12),
        (11, 12), (5, 11), (6, 12), (5, 6),
        (5, 7), (6, 8), (7, 9), (8, 10),
        (1, 2), (0, 1), (0, 2), (1, 3),
        (2, 4), (3, 5), (4, 6)
    )
}


class BaseEstimator(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def estimation(self,
                   image: Union[str, np.ndarray]
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        an abstract method need to be implemented.

        Params:
            image(str, numpy.ndarray): 图像文件路径 or 图像数据(BGR);

        Returns:
            landmarks(numpy.ndarray): 检测得到的人体关键点, shape(num_human, num_kpts, d);

        """
        pass

    def draw_info(self,
                  image: np.ndarray,
                  landmarks: np.ndarray,
                  skeletons: dict = COCO_BODY_17,
                  line_color: tuple = (255, 0, 0),
                  point_color: tuple = (0, 0, 255)
                  ) -> np.ndarray:
        """
        Draw landmarks info.

        Params:
            image(numpy.ndarray): 图像数据, shape(H, W, 3), color(B, G, R), dtype"uint8";
            landmarks(numpy.ndarray): 检测得到的人体关键点, shape(num_human, num_kpts, d);

        Returns:
            image_show(numpy.ndarray)

        """
        image_h, image_w = image.shape[:2]
        th = round(0.002 * (image_h + image_w) / 2) + 1
        for landmark in landmarks:
            # for a single person
            for part_name, skeleton in skeletons.items():
                # draw lines
                for st_idx, ed_idx in skeleton:
                    st_point = landmark[st_idx].astype("int32")
                    ed_point = landmark[ed_idx].astype("int32")
                    if st_point[0] == -1 or st_point[1] == -1 or \
                            ed_point[0] == -1 or ed_point[1] == -1:
                        continue
                    cv2.line(image, st_point, ed_point, color=line_color,
                             thickness=th, lineType=cv2.LINE_AA)
                # draw points
                for st_idx, ed_idx in skeleton:
                    st_point = landmark[st_idx].astype("int32")
                    ed_point = landmark[ed_idx].astype("int32")
                    cv2.circle(image, st_point, radius=th*2, color=point_color,
                               thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(image, ed_point, radius=th*2, color=point_color,
                               thickness=-1, lineType=cv2.LINE_AA)
        return image
