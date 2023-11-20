import os

import cv2
import ffmpeg
import numpy as np
import torch
from torchvision import transforms as T
from tqdm import tqdm

from modules.base.base_matting import BaseMatting
from utils.image_utils import *
from utils.video_utils import *


# ############################################################
# Background Matting
# ############################################################
class BackGroundMatting(BaseMatting):
    def __init__(self,
                 model: str,
                 device: str = "cpu",
                 gpu_id: int = -1
                 ) -> None:
        super().__init__()
        # Device
        if device.lower() == "cpu":
            self.device = torch.device("cpu")
        elif device.lower() == "cuda":
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            raise ValueError("device Error.")

        # Precision
        self.precision = torch.float32

        # Model
        assert os.path.exists(model), \
            "Can not find the mode weight at:{}".format(model)
        self.model = torch.jit.load(model)
        self.model.backbone_scale = 0.25
        self.model.refine_mode = "sampling"
        self.model.refine_sample_pixels = 80_000
        self.model = self.model.to(self.device)

    def image_matting(self,
                      image: Union[str, np.ndarray],
                      background_ref: Union[str, tuple, list, np.ndarray]
                      ) -> np.ndarray:
        """
        Params:
            image(str, np.ndarray): image path or image ndarray.
            background_ref(str, tuple, list, np.ndarray): reference background.
        """

        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype="uint8"), 1)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

        bg_ref = get_image(background_ref, height=h, width=w)

        alpha_ref = T.ToTensor()(bg_ref.astype("float32")/255.)
        alpha_ref = alpha_ref.unsqueeze(0)
        alpha_ref = alpha_ref.to(self.precision).to(self.device)

        input_tensor = T.ToTensor()(image_rgb.copy())
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.precision).to(self.device)

        pha, fgr, _, _, _, _ = self.model(input_tensor, alpha_ref)
        pha = pha.detach().cpu().numpy()[0, 0, ...]
        alpha_pred = (pha*255).astype("uint8")

        return alpha_pred

    def video_matting(self,
                      input_video_path: str,
                      background_ref: Union[str, tuple,
                                            list, np.ndarray] = "green",
                      output_alpha_path: str = "",
                      background_trt: str = None,
                      output_video_path: Union[str, tuple,
                                               list, np.ndarray] = None,
                      ) -> None:
        """
        Params:
            input_video_path(str): input video path.
            background_ref(str, tuple, list, np.ndarray): reference background.
            output_alpha_path(str): output alpha path, frame is like black/white.
            background_trt(str, tuple, list, np.ndarray): target background.
            output_video_path(str): output video path, 
                frame's foreground is input video, background is background_trt.
        """

        assert not (output_video_path is None and output_alpha_path is None), \
            "Please make sure that `output_video_path` and `output_alpha_path` are not all empty."

        video_reader = VideoReader(video_read_path=input_video_path)
        h, w = video_reader.shape
        # reference background
        bg_ref = get_image(background_ref, height=h, width=w)

        if output_alpha_path is not None:
            alpha_writer = VideoWriter(video_write_path=output_alpha_path,
                                       height=h, width=w,
                                       fps=video_reader.fps,
                                       pixel_format="rgb24")
        else:
            alpha_writer = None

        if output_video_path is not None:
            # target background
            bg_trt = get_image(background_trt, height=h, width=w)
            video_writer = VideoWriter(video_write_path=output_video_path,
                                       height=h, width=w,
                                       fps=video_reader.fps,
                                       pixel_format="rgb24",
                                       audio=ffmpeg.input(input_video_path).audio)
        else:
            video_writer = None

        pbar = tqdm(total=len(video_reader))
        for frame_idx in range(len(video_reader)):
            frame_bgr = video_reader[frame_idx]
            frame_rgb = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)

            alpha_pred = self.image_matting(frame_bgr, background_ref=bg_ref)
            alpha_frame = alpha_pred.astype("uint8")
            alpha_frame = np.concatenate(
                [alpha_frame[:, :, np.newaxis],]*3, axis=-1)

            if alpha_writer is not None:
                alpha_writer.write(alpha_frame)
            if video_writer is not None:
                image_rgb = frame_rgb.copy()
                frame_out = (image_rgb * (alpha_frame/255.) +
                             bg_trt * (1-alpha_frame/255.)).astype("uint8")
                video_writer.write(frame_out)
            pbar.update(1)

        pbar.close()
        video_reader.close()
        if alpha_writer is not None:
            alpha_writer.close()
        if video_writer is not None:
            video_writer.close()
