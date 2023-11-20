from typing import Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from modules.base.base_estimator import BaseEstimator


class Yolov7_Estimator(BaseEstimator):
    def __init__(self,
                 model: str,
                 device: str = "cpu",
                 gpu_id: int = -1,
                 conf_threshold: float = 0.3
                 ) -> None:
        super().__init__()
        # Params
        self.conf_threshold = conf_threshold
        self.class_names = ["people"]
        # Device
        if device.lower() == "cpu":
            self.device = ["CPUExecutionProvider"]
        elif device.lower() == "cuda":
            self.device = [
                ("CUDAExecutionProvider", {"device_id": str(gpu_id)}),
            ]
        else:
            raise ValueError(f"Unknown device: {device}.")
        # Model
        self.model = ort.InferenceSession(model, providers=self.device)

        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape[-1]

    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = (self.input_shape, self.input_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better val mAP)
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def pre_process(self, input_image: np.ndarray):
        """
        图像数据预处理.

        Params:
            input_image(numpy.ndarray): 图像数据, shape(H, W, 3), color(B, G, R), dtype"uint8";
        """
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = np.ascontiguousarray(image, dtype="float32")
        image = image / 255.
        output_image = image.transpose(2, 0, 1)[np.newaxis,]

        return output_image, ratio, dwdh

    def post_process(self, output, ratio, dwdh):
        det_bboxes = output[:, 0:4]
        det_scores = output[:, 4]
        det_labels = output[:, 5]
        det_kpts = output[:, 6:]

        bboxes = []
        scores = []
        landmarks = []
        for idx in range(len(det_bboxes)):
            bbox = det_bboxes[idx].copy()
            bbox = (bbox - dwdh * 2) / ratio
            bboxes.append(bbox)

            score = det_scores[idx]
            scores.append([score])

            kpts = det_kpts[idx].reshape(-1, 3)
            kpts[:, :2] = kpts[:, :2] - np.array(list(dwdh))
            kpts[:, :2] = kpts[:, :2] / ratio
            neg_idxs = np.where(kpts[:, -1] < self.conf_threshold)[0]
            kpts[neg_idxs, :2] = -1
            kpts = kpts[:, :2].astype("float32")
            landmarks.append(kpts)

        bboxes = np.array(bboxes, dtype="float32")
        scores = np.array(scores, dtype="float32")
        landmarks = np.array(landmarks, dtype="float32")

        return bboxes, scores, landmarks

    def estimation(self,
                   image: Union[str, np.ndarray]
                   ) -> np.ndarray:

        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype="uint8"), 1)

        input_image, ratio, dwdh = self.pre_process(image)
        outputs = self.model.run([], {self.input_name: input_image})[0]

        bboxes, scores, landmarks = self.post_process(outputs, ratio, dwdh)

        return bboxes, scores, landmarks
