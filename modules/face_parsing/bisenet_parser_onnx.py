import math
import os
from typing import Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from modules.base.base_parser import BaseParser


class BiseNet_FaceParser(BaseParser):
    def __init__(self,
                 model: str,
                 device: str = "cpu",
                 gpu_id: int = -1
                 ) -> None:
        super().__init__()
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

        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std = (0.229, 0.224, 0.225)
        self.input_size = self.model.get_inputs()[0].shape[-1]
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

        self.class_names = [
            "skin", "l_brow", "r_brow", "l_eye", "r_eye",
            "eye_g", "l_ear", "r_ear", "ear_r", "nose",
            "mouth", "u_lip", "l_lip", "neck", "neck_l",
            "cloth", "hair", "hat"
        ]
        self.part_dict = {
            "head": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18),
            "neck": (14, 15),
            "torso": (16, )
        }

    def prepare(self, input_image: np.ndarray):
        """
        图像数据预处理.

        Params:
            input_image(numpy.ndarray): 图像数据, shape(H, W, 3), color(B, G, R), dtype"uint8";
        """
        # resize
        image_bgr = cv2.resize(input_image, (self.input_size, self.input_size),
                               interpolation=cv2.INTER_LINEAR)
        # bgr to rgb
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb, dtype="float32")
        # normalization
        mean = np.array(self.input_mean).reshape(1, -1).astype("float64")
        stdinv = 1 / np.array(self.input_std).reshape(1, -1).astype("float64")
        mean *= 255
        stdinv /= 255
        image_norm = cv2.subtract(image_rgb, mean)
        image_norm = cv2.multiply(image_norm, stdinv)
        output_image = image_norm.transpose(2, 0, 1)[np.newaxis,]

        return output_image

    def parsing(self,
                image: Union[str, np.ndarray]
                ) -> np.ndarray:
        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype="uint8"), 1)

        origin_h, origin_w = image.shape[:2]
        # preprocess
        input_image = self.prepare(image)
        # inference
        outputs = self.model.run([self.output_name,],
                                 {self.input_name: input_image})
        # postprocess
        pred_label = outputs[0].squeeze(0).argmax(0).astype("uint8")
        label = cv2.resize(pred_label, (origin_w, origin_h),
                           cv2.INTER_NEAREST)
        return label

    def get_head_mask(self, label: np.ndarray):
        h, w = label.shape[:2]
        mask = np.ones((h, w, 3), dtype="uint8") * (255, 255, 255)
        head_coords = np.where(
            (label == 1) | (label == 2) | (label == 3) |
            (label == 4) | (label == 5) | (label == 6) |
            (label == 7) | (label == 8) | (label == 9) |
            (label == 10) | (label == 11) | (label == 12) |
            (label == 13) | (label == 17) | (label == 18)
        )
        mask[head_coords[0], head_coords[1], :] = (255, 0, 0)
        neck_coords = np.where((label == 14) | (label == 15))
        mask[neck_coords[0], neck_coords[1], :] = (0, 255, 0)
        torso_coords = np.where((label == 16))
        mask[torso_coords[0], torso_coords[1], :] = (0, 0, 255)

        return mask
