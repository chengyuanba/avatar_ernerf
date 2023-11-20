import os
from typing import Union

import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm

from utils.video_utils import VideoReader, VideoWriter


class MergeProcessor(object):
    def __init__(self) -> None:
        self.mode_dict = {
            0: self.fusion_head_bbox,
            1: self.fusion_face_area,
            2: self.fusion_face_neck,
        }

    def fusion_head_bbox(self,
                         origin_frame_bgr: np.ndarray,
                         head_region: Union[tuple, list, np.ndarray],
                         head_label: np.ndarray = None,
                         head_frame_rgb: np.ndarray = None,
                         head_frame_depth: np.ndarray = None,
                         ):
        """
        Params:
            origin_frame_bgr(np.ndarray): 原始视频的图像帧(全身);
            head_region(tuple, list, np.ndarray): 头部区域坐标, (x1, y1, x2, y2);
            head_label(np.ndarray): 人脸解析mask标签(头部);
            head_frame_rgb(np.ndarray): 推理结果的图像帧(头部, rgb图);
            head_frame_depth(np.ndarray): 推理结果的图像帧(头部, depth图);

        """
        x1, y1, x2, y2 = head_region
        pred_head = cv2.resize(
            head_frame_rgb, (int(x2-x1), int(y2-y1)),
            interpolation=cv2.INTER_LINEAR
        )
        pred_head = cv2.cvtColor(pred_head, cv2.COLOR_RGB2BGR)
        origin_frame_bgr[y1:y2, x1:x2, :] = pred_head

        return origin_frame_bgr

    def fusion_face_area(self,
                         origin_frame_bgr: np.ndarray,
                         head_region: Union[tuple, list, np.ndarray],
                         head_label: np.ndarray = None,
                         head_frame_rgb: np.ndarray = None,
                         head_frame_depth: np.ndarray = None,
                         ):
        """
        Params:
            origin_frame_bgr(np.ndarray): 原始视频的图像帧(全身);
            head_region(tuple, list, np.ndarray): 头部区域坐标, (x1, y1, x2, y2);
            head_label(np.ndarray): 人脸解析mask标签(头部);
            head_frame_rgb(np.ndarray): 推理结果的图像帧(头部, rgb图);
            head_frame_depth(np.ndarray): 推理结果的图像帧(头部, depth图);

        """
        x1, y1, x2, y2 = head_region
        orig_head = origin_frame_bgr[y1:y2, x1:x2, :]
        pred_head = cv2.resize(
            head_frame_rgb, (int(x2-x1), int(y2-y1)),
            interpolation=cv2.INTER_LINEAR
        )
        pred_head = cv2.cvtColor(pred_head, cv2.COLOR_RGB2BGR)
        head_label = cv2.resize(
            head_label, (int(x2-x1), int(y2-y1)),
            interpolation=cv2.INTER_NEAREST
        )
        face_mask_index = np.where(
            (head_label == 1) | (head_label == 2) | (head_label == 3) |
            (head_label == 4) | (head_label == 5) | (head_label == 6) |
            (head_label == 10) | (head_label == 11) | (head_label == 12) |
            (head_label == 13)
        )
        face_mask = np.zeros(shape=(y2-y1, x2-x1), dtype="int32")
        face_mask[face_mask_index[0], face_mask_index[1]] = 255
        face_neck_mask_1c = np.clip(
            face_mask, a_min=0, a_max=255).astype("int32")
        face_neck_mask_3c = np.concatenate(
            [face_neck_mask_1c[:, :, np.newaxis], ]*3,
            axis=2).astype("float32")
        merge_head = orig_head * (1 - face_neck_mask_3c/255.) + \
            pred_head * (face_neck_mask_3c/255.)
        origin_frame_bgr[y1:y2, x1:x2, :] = merge_head

        return origin_frame_bgr

    def fusion_face_neck(self,
                         origin_frame_bgr: np.ndarray,
                         head_region: Union[tuple, list, np.ndarray],
                         head_label: np.ndarray = None,
                         head_frame_rgb: np.ndarray = None,
                         head_frame_depth: np.ndarray = None,
                         ):
        """
        Params:
            origin_frame_bgr(np.ndarray): 原始视频的图像帧(全身);
            head_region(tuple, list, np.ndarray): 头部区域坐标, (x1, y1, x2, y2);
            head_label(np.ndarray): 人脸解析mask标签(头部);
            head_frame_rgb(np.ndarray): 推理结果的图像帧(头部, rgb图);
            head_frame_depth(np.ndarray): 推理结果的图像帧(头部, depth图);

        """
        x1, y1, x2, y2 = head_region
        orig_head = origin_frame_bgr[y1:y2, x1:x2, :]
        pred_head = cv2.resize(
            head_frame_rgb, (int(x2-x1), int(y2-y1)),
            interpolation=cv2.INTER_LINEAR
        )
        pred_head = cv2.cvtColor(pred_head, cv2.COLOR_RGB2BGR)
        head_label = cv2.resize(
            head_label, (int(x2-x1), int(y2-y1)),
            interpolation=cv2.INTER_NEAREST
        )
        face_neck_mask_index = np.where(
            (head_label == 1) | (head_label == 2) | (head_label == 3) |
            (head_label == 4) | (head_label == 5) | (head_label == 6) |
            (head_label == 10) | (head_label == 11) | (head_label == 12) |
            (head_label == 13) | (head_label == 14) | (head_label == 15)
        )
        face_neck_mask = np.zeros(shape=(y2-y1, x2-x1), dtype="int32")
        face_neck_mask[face_neck_mask_index[0], face_neck_mask_index[1]] = 255
        face_neck_mask_1c = np.clip(
            face_neck_mask, a_min=0, a_max=255).astype("int32")
        face_neck_mask_3c = np.concatenate(
            [face_neck_mask_1c[:, :, np.newaxis], ]*3,
            axis=2).astype("float32")
        merge_head = orig_head * (1 - face_neck_mask_3c/255.) + \
            pred_head * (face_neck_mask_3c/255.)
        origin_frame_bgr[y1:y2, x1:x2, :] = merge_head

        return origin_frame_bgr

    def run(self,
            model_config: dict,
            audio_path: str,
            save_path: str,
            head_frame_rgb_list: list,
            head_frame_depth_list: list,
            frame_index_start: int,
            frame_index_end: int,
            background_rgb: tuple = None,
            with_alpha: bool = False,
            mode: int = 0):
        """
        Params:
            model_data_cfg(dict): 模特属性配置;
            audio_path(str): 输入音频文件路径;
            save_path(str): 输出视频文件保存路径;
            head_frame_rgb_list(list): 推理结果的头部rgb图像;
            head_frame_depth_list(list): 推理结果的头部深度图;
            frame_index_start(int): 使用训练数据的起始帧索引;
            frame_index_end(int): 使用训练数据的终止帧索引;
            with_alpha(bool): 是否返回含有alpha通道的视频;
            mode(int): 后处理模式;
        """

        input_video_path = os.path.join(
            model_config["data_dir"], model_config["video_file"])
        alpha_video_path = os.path.join(
            model_config["data_dir"], model_config["alpha_file"])
        face_label_data_dir = os.path.join(
            model_config["data_dir"], "parsing_label")
        x1, y1, x2, y2 = model_config["head_region"]

        input_video = VideoReader(input_video_path)
        frame_height = int(input_video.shape[0])
        frame_width = int(input_video.shape[1])
        fps = float(input_video.fps)

        if background_rgb is not None:
            background_bgr = np.array(background_rgb[::-1], dtype="uint8")
            background_bgr = np.array(background_bgr, dtype="uint8")
            background = np.ones(
                shape=(frame_height, frame_width, 3), dtype="uint8")
            background = (background * background_bgr).astype("uint8")
        if with_alpha:
            input_alpha = VideoReader(alpha_video_path)
        frame_count = len(head_frame_rgb_list)
        temp_frame_idxs = list(range(frame_index_start, frame_index_end))
        select_frame_idxs = temp_frame_idxs + temp_frame_idxs[::-1]
        while len(select_frame_idxs) < frame_count:
            select_frame_idxs *= 2
        select_frame_idxs = select_frame_idxs[:frame_count]
        output_video = VideoWriter(
            save_path,  height=frame_height,
            width=frame_width, fps=fps,
            pixel_format="rgba" if with_alpha else "rgb24",
            audio=ffmpeg.input(audio_path).audio
        )

        for real_idx, frame_idx in tqdm(enumerate(select_frame_idxs), total=len(select_frame_idxs)):
            input_frame = np.array(input_video[frame_idx])
            output_frame = input_frame.copy()
            if with_alpha:
                alpha_frame = np.array(input_alpha[frame_idx])
            head_frame_rgb = np.array(
                head_frame_rgb_list[real_idx], dtype="uint8")
            # head_frame_depth = head_frame_depth_list[real_idx]
            head_label_file = os.path.join(
                face_label_data_dir, f"{frame_idx}.png")
            head_label = cv2.imread(head_label_file)
            head_label = np.array(head_label, dtype="uint8")

            output_frame = self.mode_dict[mode](
                origin_frame_bgr=output_frame,
                head_region=(x1, y1, x2, y2),
                head_label=head_label,
                head_frame_rgb=head_frame_rgb,
            )
            if with_alpha:
                if background_rgb is not None:
                    output_frame = output_frame - \
                        (1.-alpha_frame/255.) * background
                    output_frame = np.clip(
                        output_frame, 0, 255).astype("uint8")
                    output_frame = output_frame * (alpha_frame/255.) + \
                        background * (1.-alpha_frame/255.)
                alpha_frame = self.cv_tools.convert_color(
                    mode="bgr2gray", data_cupy=alpha_frame)

                output_frame = np.concatenate(
                    [output_frame, np.expand_dims(alpha_frame, axis=-1)],
                    axis=-1
                ).astype("uint8")

            if with_alpha:
                frame_rgba = output_frame[:, :, [2, 1, 0, 3]]
                output_video.write(frame_rgba)
            else:
                frame_rgb = output_frame[:, :, [2, 1, 0]]
                output_video.write(frame_rgb)

        input_video.close()
        if with_alpha:
            input_alpha.close()
        output_video.close()

        return
