import abc
from typing import Tuple, Union

import cv2
import numpy as np


class BaseFaceDetector(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def detection(self,
                  image: Union[str, np.ndarray]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        an abstract method need to be implemented.

        Params:
            image(str, numpy.ndarray): 图像文件路径 or 图像数据(BGR);

        Returns:
            BBoxes(numpy.ndarray): 检测得到的人脸边框, shape(n, 4), [x1, y1, x2, y2];
            Scores(numpy.ndarray): 每个人脸边框对应的置信度, shape(n, 1);
            Keypoints(numpy.ndarray): 人脸关键点数据, shape(n, 5, d);

        """
        pass

    def draw_info(self,
                  image: np.ndarray, bboxes: np.ndarray,
                  scores: np.ndarray, keypoints: np.ndarray,
                  bbox_color: tuple = (255, 0, 0),
                  text_color: tuple = (225, 255, 255),
                  point_color: tuple = (225, 0, 255)
                  ) -> np.ndarray:
        """
        Draw detection info.

        Params:
            image(numpy.ndarray): 图像数据, shape(H, W, 3), color(B, G, R), dtype"uint8";
            BBoxes(numpy.ndarray): 检测得到的人脸边框, shape(n, 4), [x1, y1, x2, y2];
            Scores(numpy.ndarray): 每个人脸边框对应的置信度, shape(n, 1);
            Keypoints(numpy.ndarray): 人脸关键点数据, shape(n, 5, d);

        Returns:
            image_show(numpy.ndarray)

        """
        image_h, image_w = image.shape[:2]
        th01 = round(0.002 * (image_h + image_w) / 2) + 1
        th02 = max(th01 - 1, 1)
        for bbox, score, kpt in zip(bboxes, scores, keypoints):
            # draw bbox
            c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(image, c1, c2, color=bbox_color,
                          thickness=th01, lineType=cv2.LINE_AA)
            # draw score
            score = round(score[0], 4)
            text_size = cv2.getTextSize(str(score), 0, fontScale=th01/3,
                                        thickness=th02)[0]
            c2 = (c1[0]+text_size[0], c1[1]-text_size[1]-3)
            cv2.rectangle(image, c1, c2, color=bbox_color,
                          thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(image, str(score), (c1[0], c1[1]-2), 0,
                        th01/3, color=text_color,
                        thickness=th02, lineType=cv2.LINE_AA)
            # draw keypoint
            for pidx in range(len(kpt)):
                point = kpt[pidx, :2].astype("int")
                cv2.circle(image, point, radius=2, color=point_color,
                           thickness=-1, lineType=cv2.LINE_AA)
        return image
