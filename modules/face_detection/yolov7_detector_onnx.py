from typing import Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from modules.base.base_face_detector import BaseFaceDetector


class Yolov7_FaceDetector(BaseFaceDetector):
    def __init__(self,
                 model: str,
                 device: str = "cpu",
                 gpu_id: int = -1,
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.5
                 ) -> None:
        super().__init__()
        # Params
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.class_names = ["face"]
        self.num_classes = len(self.class_names)
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

        model_inputs = self.model.get_inputs()
        self.input_names = [model_inputs[i].name
                            for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        model_outputs = self.model.get_outputs()
        self.output_names = [model_outputs[i].name
                             for i in range(len(model_outputs))]

    def resize_image(self, input_image, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and input_image.shape[0] != input_image.shape[1]:
            hw_scale = input_image.shape[0] / input_image.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(
                    self.input_width / hw_scale)
                img = cv2.resize(input_image, (neww, newh),
                                 interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width-neww-left,
                                         cv2.BORDER_CONSTANT, value=(114, 114, 114))
            else:
                newh, neww = int(self.input_height *
                                 hw_scale), self.input_width
                img = cv2.resize(input_image, (neww, newh),
                                 interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height-newh-top, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(114, 114, 114))
        else:
            img = cv2.resize(input_image, (self.input_width, self.input_height),
                             interpolation=cv2.INTER_AREA)

        return img, newh, neww, top, left

    def pre_process(self, image):
        self.img_height, self.img_width = image.shape[:2]
        self.scale = np.array([self.img_width / self.input_width,
                               self.img_height / self.input_height,
                               self.img_width / self.input_width,
                               self.img_height / self.input_height],
                              dtype=np.float32)
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(
            input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :]

        return input_tensor

    def post_process(self, output):
        predictions = np.squeeze(output[0]).reshape((-1, 21))

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5] *= obj_conf

        # Get the scores
        scores = predictions[:, 5]

        # Filter out the objects with a low score
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]

        # Get bounding boxes for each object
        boxes, kpts = self.extract_boxes(predictions)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold,
                                   self.iou_threshold)
        return boxes[indices], scores[indices], kpts[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4] * self.scale
        kpts = predictions[:, 6:]  # x1,y1,score1, ...., x5,y5,score5
        kpts *= np.tile(np.array([self.scale[0],
                        self.scale[1], 1], dtype=np.float32), (1, 5))

        # Convert boxes to xywh format
        boxes_ = np.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        return boxes_, kpts

    def detection(self,
                  image: Union[str, np.ndarray]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype="uint8"), 1)

        input_image = self.pre_process(image)

        outputs = self.model.run(self.output_names,
                                 {input_name: input_image
                                  for input_name in self.input_names})
        det_bboxes, det_conf, landmarks = self.post_process(outputs)

        det_bboxes[:, 2] = det_bboxes[:, 0] + det_bboxes[:, 2]
        det_bboxes[:, 3] = det_bboxes[:, 1] + det_bboxes[:, 3]

        bboxes = det_bboxes.reshape(-1, 4)
        scores = det_conf.reshape(-1, 1)
        keypoints = landmarks.reshape(-1, 5, 3)

        return bboxes, scores, keypoints
