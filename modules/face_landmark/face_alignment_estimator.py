from typing import Tuple, Union

import cv2
import numpy as np
from face_alignment import FaceAlignment, LandmarksType

from modules.base.base_estimator import BaseEstimator

FACE_SKELETON = {
    "face_oval": ((0, 1), (1, 2), (2, 3), (3, 4),
                  (4, 5), (5, 6), (6, 7), (7, 8),
                  (8, 9), (9, 10), (10, 11), (11, 12),
                  (12, 13), (13, 14), (14, 15), (15, 16)),
    "left_eyebrow": ((17, 18), (18, 19), (19, 20), (20, 21)),
    "right_eyebrow": ((22, 23), (23, 24), (24, 25), (25, 26)),
    "nose": ((27, 28), (28, 29), (29, 30), (30, 33),
             (33, 32), (32, 31), (33, 34), (34, 35)),
    "left_eye": ((36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)),
    "right_eye": ((42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)),
    "lips_outer": ((48, 49), (49, 50), (50, 51), (51, 52),
                   (52, 53), (53, 54), (54, 55), (55, 56),
                   (56, 57), (57, 58), (58, 59), (59, 48)),
    "lips_inner": ((60, 61), (61, 62), (62, 63), (63, 64),
                   (64, 65), (65, 66), (66, 67), (67, 60)),
}


class FaceAlignment_Estimator(BaseEstimator):
    def __init__(self,
                 model: str = "sfd",
                 device: str = "cpu",
                 gpu_id: int = -1
                 ) -> None:
        super().__init__()
        assert model in ["dlib", "sfd", "blazeface"]
        # Device
        self.device = device
        # Model
        self.model = FaceAlignment(
            landmarks_type=LandmarksType.TWO_D,
            face_detector=model, device=device,
            flip_input=False
        )

    def estimation(self,
                   image: Union[str, np.ndarray]
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype="uint8"), 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = self.model.get_landmarks(image_rgb)

        return None, None, np.array(landmarks, dtype="float32")
