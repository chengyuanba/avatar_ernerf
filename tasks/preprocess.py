"""
输入数据前处理

Task 01: Video Format Conversion
Task 02: Video Matting
Task 03: Save Preview Image
Task 04: Head Region Detection and Crop & Resize
Task 05: Audio Stream Extraction
Task 06: Audio Feature Extraction
Task 07: Video to Images
Task 08: Run Face Parsing
Task 09: Run Face Landmark
Task 10: Run Background Extraction
Task 11: Run Torso and Ground Truth Extraction
Task 12: Run OpenFace FeatureExtraction
Task 13: Run Face Tracking
Task 14: Save `transforms.json`
"""

import glob
import json
import logging
import os
import shutil
import time
from typing import Tuple, Union

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from modules.audio_features.encoding_with_transformers import audio_encoding
from modules.face_detection.yolov7_detector_onnx import Yolov7_FaceDetector
from modules.face_landmark.face_alignment_estimator import \
    FaceAlignment_Estimator
from modules.face_parsing.bisenet_parser_onnx import BiseNet_FaceParser
from modules.face_tracking.face_tracker_new import face_tracker_training
from modules.human_pose_estimation.yolov7_estimator_onnx import \
    Yolov7_Estimator
from modules.matting.bgm_torchscript import BackGroundMatting
from utils.video_utils import VideoReader, get_video_seconds

# ############################################################
# Model Weights
# ############################################################
pretrained_weights_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pretrained_weights")
# =====>> Face Detection
face_det_yolov7_weight = os.path.join(
    pretrained_weights_dir, "face_detection", "yolov7-tiny-face.onnx")
# =====>> Face Parsing
face_seg_bisenet_weight = os.path.join(
    pretrained_weights_dir, "face_parsing", "bisenet-face.onnx")
# =====>> Human Pose Estimation
human_pose_yolov7_weight = os.path.join(
    pretrained_weights_dir, "pose_estimation", "yolov7-w6-pose-nms.onnx")
# =====>> Matting
matting_bgm_weight = os.path.join(
    pretrained_weights_dir, "matting", "bgm_torchscript_resnet101_fp32.pth")
# =====>> ThreeDMM
ThreeDMM_weight = os.path.join(pretrained_weights_dir, "3DMM")


def video_format_conversion(input_path: str,
                            output_path: str,
                            quality: int = 18,
                            target_fps: float = 25,
                            max_seconds: float = 600
                            ) -> None:
    """视频转换, 输入视频会被转换为25fps, 最大时长600秒, libx264格式.

    Params:
        input_path(str):  输入视频路径;
        output_path(str): 输出视频路径, 格式为`.mp4`;
        quality(int): 输出视频的质量, 数值越高, 视频文件越大, 默认为18;
        target_fps(float): 目标帧率, 默认为25;
        max_seconds(float): 最大时长, 默认为600(单位:秒);
    """

    assert os.path.exists(input_path), \
        logging.error("The input video path does not exist.")
    if os.path.splitext(output_path)[1] != ".mp4":
        logging.warning("The save path is not a `.mp4` file, "
                        "the suffix will be changed to `.mp4`.")
        output_path = os.path.splitext(output_path)[0] + ".mp4"

    total_seconds = get_video_seconds(input_path)

    cmd = "ffmpeg -loglevel error "
    if total_seconds > max_seconds:
        logging.info("The video is too long, "
                     "we will cut the first {} seconds.".format(max_seconds))
        cmd += "-ss 00:00:00 -t 00:10:00 "
    cmd += "-i {} -vcodec libx264 -crf {} -r {}  -y {}".format(
        input_path, quality, target_fps, output_path)
    logging.info("[Commond]:\n{}".format(cmd))
    os.system(cmd)

    logging.info("Save the converted video to: {}".format(output_path))


def video_matting(input_path: str,
                  background_ref: Union[str, tuple,
                                        list, np.ndarray] = "green",
                  output_alpha_path: str = None,
                  background_trt: Union[str, tuple, list, np.ndarray] = "gray",
                  output_video_path: str = None,
                  device: str = "cpu",
                  gpu_id: int = -1
                  ) -> None:
    """视频背景抠图.

    Params:
        input_path(str): 输入视频路径;
        background_ref(str, tuple, list, np.ndarray): 参考背景图;
        output_alpha_path(str): 输出抠图结果的视频路径, 格式为`.mp4`;
        background_trt(str, tuple, list, np.ndarray): 目标背景图;
        output_alpha_path(str): 输出替换背景后的视频路径, 格式为`.mp4`;
        device(str): device;
        gpu_id(int): gpu index;
    """
    assert os.path.exists(input_path), \
        logging.error("The input video path does not exist.")

    matting_tool = BackGroundMatting(
        model=matting_bgm_weight,
        device=device, gpu_id=gpu_id
    )

    matting_tool.video_matting(
        input_video_path=input_path,
        background_ref=background_ref,
        output_alpha_path=output_alpha_path,
        background_trt=background_trt,
        output_video_path=output_video_path,
    )

    logging.info("Save the target  video to: {}".format(output_video_path))
    logging.info("Save the matting video to: {}".format(output_alpha_path))


def save_preview_image(input_path: str,
                       output_path: str
                       ) -> None:
    """
    Params:
        input_path(str): 输入视频路径;
        output_dir(str): 首帧图像保存路径;
    """
    assert os.path.exists(input_path), \
        logging.error("The input video path does not exist.")

    video_reader = VideoReader(input_path)
    first_frame = video_reader[0]
    video_reader.close()

    cv2.imwrite(output_path, first_frame)
    logging.info("Save preview image to: {}".format(output_path))


def head_region_detection(input_path: str,
                          alpha_path: str,
                          output_path: str,
                          only_first: bool = False,
                          skip_steps: int = 24,
                          quality: int = 18,
                          target_size: int = 512,
                          device: str = "cpu",
                          gpu_id: int = -1
                          ) -> Tuple[int, int, int, int]:
    """
    Params:
        input_path(str): 输入视频路径;
        alpha_path(str): 抠图视频路径;
        output_path(str): 输出视频路径, 格式为`.mp4`;
        only_first(bool): 是否仅首帧做参考, 该模式下头部区域的坐标仅参考视频的首帧, `skip_steps`不再生效;
        skip_steps(int): 所选择的每连续的两帧参考帧之间跳过的帧数, 默认为24;
        quality(int): 输出视频的质量, 数值越高, 视频文件越大, 默认为18;
        target_size(int): 输出视频的尺寸, 默认为512;
        device(str): device;
        gpu_id(int): gpu index;

    Returns:
        head_bbox(tuple of int): 人像头部区域, 格式为(x1, y1, x2, y2);
    """
    assert os.path.exists(input_path), \
        logging.error("The input video path does not exist.")

    if os.path.splitext(output_path)[1] != ".mp4":
        logging.warning("The save path is not a `.mp4` file, "
                        "the suffix will be changed to `.mp4`.")
        output_path = os.path.splitext(output_path)[0] + ".mp4"

    # 初始化人脸检测器, 姿态估计器, 资源视频读取器, 抠图视频读取器
    face_detector = Yolov7_FaceDetector(
        model=face_det_yolov7_weight,
        device=device, gpu_id=gpu_id
    )
    pose_estimator = Yolov7_Estimator(
        model=human_pose_yolov7_weight,
        device=device, gpu_id=gpu_id
    )
    video_reader = VideoReader(input_path)
    alpha_reader = VideoReader(alpha_path)

    # 首帧模式
    if only_first:
        logging.info("Using `only_first` mode.")
        image_frame = video_reader[0]
        alpha_frame = alpha_reader[0]

        # 通过人脸检测和姿态估计, 获取face_bbox和body_landmarks
        face_bboxes, face_bbox_scores, face_kpts = face_detector.detection(
            image_frame.copy())
        _, _, multi_body_landmarks = pose_estimator.estimation(
            image_frame.copy())

        # 通过抠图结果, 获取人像最上侧边缘位置
        alpha_gray = cv2.cvtColor(alpha_frame, cv2.COLOR_BGR2GRAY)
        matting_ys, matting_xs = np.where(alpha_gray > 128)
        # 通过人脸检测结果, 获取人脸bbox位置
        # [!!!Attention]保险机制: 当检测结果中出现多张人脸时, 取置信度最高的一个.
        max_score_idx = face_bbox_scores.argmax()
        face_bbox = face_bboxes[max_score_idx]
        face_5kpt = face_kpts[max_score_idx].reshape(-1, 3)
        head_center = (face_5kpt[:, 0].mean(), face_5kpt[:, 1].mean())
        # 通过人体姿态估计结果, 获取人体关键点位置
        # [!!!Attention]仅选择第0个
        body_landmarks = multi_body_landmarks[0]

        left_ear_kpt = body_landmarks[3, :2]
        right_ear_kpt = body_landmarks[4, :2]
        left_shoulder_kpt = body_landmarks[5, :2]
        right_shoulder_kpt = body_landmarks[6, :2]

        xmin_01 = round(min(left_ear_kpt[0], right_ear_kpt[0]))
        xmax_01 = round(max(left_ear_kpt[0], right_ear_kpt[0]))
        ymin_01 = round(np.min(matting_ys))
        ymax_01 = round(np.max((
            face_bbox[1], face_bbox[3],
            matting_ys[np.where(matting_xs == round(
                left_shoulder_kpt[0]))].min(),
            matting_ys[np.where(matting_xs == round(
                right_shoulder_kpt[0]))].min(),
        )))

        square_edge = max(abs(xmax_01 - xmin_01), abs(ymax_01 - ymin_01))
        square_edge = square_edge * (1 + 0.05)
        square_edge = round(square_edge // 2 * 2)  # 保证为偶数

        head_bbox = (
            int(head_center[0] - square_edge // 2),
            int(head_center[1] - square_edge // 2),
            int(head_center[0] + square_edge // 2),
            int(head_center[1] + square_edge // 2)
        )

        logging.info("Detected head region: {}".format(head_bbox))

    # 非首帧模式
    else:
        logging.info("Not Using `only_first` mode.")
        frame_idxs = list(range(0, len(video_reader), skip_steps+1))
        total_read_time = time.time()
        logging.info("load video file:")
        frame_list = [video_reader[idx] for idx in tqdm(frame_idxs)]
        logging.info("load alpha file:")
        alpha_list = [alpha_reader[idx] for idx in tqdm(frame_idxs)]
        total_read_time = time.time() - total_read_time

        # 逐帧进行人像头部区域检测
        pbar = tqdm(total=len(frame_idxs))
        head_bboxes = []
        head_centers = []
        total_proc_time = 0.
        for idx, (frame_idx, frame, alpha) in enumerate(zip(frame_idxs, frame_list, alpha_list)):
            proc_time_per_frame = time.time()
            alpha_gray = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
            face_bboxes, face_bbox_scores, face_kpts = face_detector.detection(
                frame.copy())
            _, _, multi_body_landmarks = pose_estimator.estimation(
                frame.copy())
            proc_time_per_frame = time.time() - proc_time_per_frame
            total_proc_time += proc_time_per_frame

            # [!!!Attention]请保证视频中有且仅有一人
            if len(multi_body_landmarks) != 1:
                raise ValueError("pose landmark error at frame index:{}, find {} people".format(
                    frame_idx, len(multi_body_landmarks)))

            matting_ys, matting_xs = np.where(alpha_gray > 128)
            max_score_idx = face_bbox_scores.argmax()
            face_bbox = face_bboxes[max_score_idx]
            face_5kpt = face_kpts[max_score_idx].reshape(-1, 3)
            head_centers.append((face_5kpt[:, 0].mean(),
                                face_5kpt[:, 1].mean()))
            body_landmarks = multi_body_landmarks[0]
            left_ear_kpt = body_landmarks[3, :2]   # 左耳
            right_ear_kpt = body_landmarks[4, :2]  # 右耳
            left_shoulder_kpt = body_landmarks[5, :2]   # 左肩
            right_shoulder_kpt = body_landmarks[6, :2]  # 右肩

            xmin_01 = round(min(left_ear_kpt[0], right_ear_kpt[0]))
            xmax_01 = round(max(left_ear_kpt[0], right_ear_kpt[0]))
            ymin_01 = round(np.min(matting_ys))
            ymax_01 = round(np.max((
                face_bbox[1], face_bbox[3],
                matting_ys[np.where(matting_xs == round(
                    left_shoulder_kpt[0]))].min(),
                matting_ys[np.where(matting_xs == round(
                    right_shoulder_kpt[0]))].min(),
            )))

            square_edge = max(abs(xmax_01 - xmin_01), abs(ymax_01 - ymin_01))
            square_edge = square_edge * (1 + 0.05)
            square_edge = round(square_edge // 2 * 2)  # 保证为偶数
            square_center = (round((xmin_01 + xmax_01) // 2),
                             round((ymin_01 + ymax_01) // 2))
            xmin_02 = square_center[0] - square_edge // 2
            xmax_02 = square_center[0] + square_edge // 2
            ymin_02 = square_center[1] - square_edge // 2
            ymax_02 = square_center[1] + square_edge // 2
            head_bboxes.append((xmin_02, ymin_02, xmax_02, ymax_02))

            pbar.update(1)
        pbar.close()

        logging.info("Total Read Time: {:.3f}ms".format(total_read_time * 1e3))
        logging.info("Read Time per frame: {:.3f}ms".format(
            total_read_time * 1e3 / len(frame_idxs)))
        logging.info("Total Proc Time: {:.3f}ms".format(total_proc_time * 1e3))
        logging.info("Proc Time per frame: {:.3f}ms".format(
            total_proc_time * 1e3 / len(frame_idxs)))

        head_bboxes = np.round(head_bboxes).astype("int32")
        head_centers = np.round(head_centers).astype("int32")

        head_bboxes_areas = (head_bboxes[:, 2] - head_bboxes[:, 0]) * \
            (head_bboxes[:, 3] - head_bboxes[:, 1])
        head_bboxes_centers = np.concatenate(
            [
                ((head_bboxes[:, 0] + head_bboxes[:, 2]) / 2).reshape(-1, 1),
                ((head_bboxes[:, 1] + head_bboxes[:, 3]) / 2).reshape(-1, 1)
            ],
            axis=-1
        )

        area_p05 = np.percentile(
            head_bboxes_areas, q=5, axis=0, keepdims=False)
        area_p95 = np.percentile(
            head_bboxes_areas, q=95, axis=0, keepdims=False)
        xc_p05, yc_p05 = np.percentile(
            head_bboxes_centers, q=5, axis=0, keepdims=False)
        xc_p95, yc_p95 = np.percentile(
            head_bboxes_centers, q=95, axis=0, keepdims=False)

        idx01 = np.where((head_bboxes_areas >= area_p05) &
                         (head_bboxes_areas <= area_p95))[0]
        idx02 = np.where((head_bboxes_centers[:, 0] >= xc_p05) &
                         (head_bboxes_centers[:, 0] <= xc_p95))[0]
        idx03 = np.where((head_bboxes_centers[:, 1] >= yc_p05) &
                         (head_bboxes_centers[:, 1] <= yc_p95))[0]
        idx04 = np.intersect1d(idx01, idx02)
        idx05 = np.intersect1d(idx03, idx04)

        bboxes_x1 = head_bboxes[idx05, 0].min()
        bboxes_y1 = head_bboxes[idx05, 1].min()
        bboxes_x2 = head_bboxes[idx05, 2].max()
        bboxes_y2 = head_bboxes[idx05, 3].max()

        bboxes_center = (
            (bboxes_x1 + bboxes_x2) / 2,
            (bboxes_y1 + bboxes_y2) / 2
        )

        bboxes_edge = min(
            bboxes_x2 - bboxes_x1,
            bboxes_y2 - bboxes_y1
        )
        bboxes_edge = int(bboxes_edge // 2 * 2)

        head_bbox = (
            int(bboxes_center[0] - bboxes_edge // 2),
            int(bboxes_center[1] - bboxes_edge // 2),
            int(bboxes_center[0] + bboxes_edge // 2),
            int(bboxes_center[1] + bboxes_edge // 2)
        )
        logging.info("Detected head region: {}".format(head_bbox))

    video_reader.close()
    alpha_reader.close()
    del face_detector
    del pose_estimator

    x1, y1, x2, y2 = head_bbox
    cmd = "ffmpeg -loglevel error -i {} ".format(input_path)
    cmd += "-vf crop={}:{}:{}:{},scale={}:{} ".format(
        int(x2-x1), int(y2-y1), int(x1), int(y1),
        int(target_size), int(target_size)
    )
    cmd += "-vcodec libx264 -crf {} -acodec aac -strict -2 -ar 16000 -ac 1 -y {}".format(
        quality, output_path)
    logging.info("[Commond]:\n{}".format(cmd))
    os.system(cmd)
    logging.info("Save the head region video to: {}".format(output_path))

    bbox_save_path = os.path.join(
        os.path.dirname(output_path), "head_bbox.json")
    with open(bbox_save_path, "w", encoding="utf-8") as f:
        json.dump({"head_bbox": head_bbox}, f)
    logging.info("Save the head bbox info to: {}".format(bbox_save_path))

    return head_bbox


def audio_stream_extraction(input_path: str,
                            output_path: str,
                            sample_rate: int = 16000,
                            mono: bool = True):
    """
    Params:
        input_path(str): 输入视频路径;
        output_path(str): 输出音频路径, 格式为`.wav`;
        sample_rate(int): 输出音频的采样率, 默认为16000;
        mono(bool): 是否单声道, 默认为True;
    """

    assert os.path.exists(input_path), \
        logging.error("The input video path does not exist.")
    if os.path.splitext(output_path)[1] != ".wav":
        logging.warning("The save path is not a `.wav` file, "
                        "the suffix will be changed to `.wav`.")
        output_path = os.path.splitext(output_path)[0] + ".wav"

    cmd = "ffmpeg -loglevel error -i {} -ac {} -ar {} -f wav -y {}".format(
        input_path, 1 if mono else 2, sample_rate, output_path)
    logging.info("[Commond]:\n{}".format(cmd))
    os.system(cmd)

    logging.info("Extracted audio stream to: {}".format(output_path))


def audio_feature_extraction(input_path: str,
                             output_path: str,
                             model_name: str = "hubert",
                             model_arch: str = "large",
                             model_weight: str = "facebook/hubert-large-ls960-ft",
                             device: str = "cpu",
                             gpu_id: int = -1,
                             ) -> Tuple[int, int, int]:
    """
    Params:
        input_path(str): 输入视频路径;
        output_path(str): 输出视频路径, 格式为`.npy`;
        model_name(str): 音频特征提取所使用的算法名称, 默认"hubert";
        model_arch(str): 音频特征提取所使用的算法结构, "base" or "large", 默认"large";
        model_weight(str): 音频特征提取所使用的算法权重, huggingface model name or local dir path,
            默认"facebook/hubert-large-ls960-ft";
        device(str): device;
        gpu_id(int): gpu index;
    """
    assert os.path.exists(input_path), \
        logging.error("The input video path does not exist.")
    if os.path.splitext(output_path)[1] != ".npy":
        logging.warning("The save path is not a `.npy` file, "
                        "the suffix will be changed to `.npy`.")
        output_path = os.path.splitext(output_path)[0] + ".npy"

    logging.info("Load audio from file: {}".format(input_path))
    feature_shape = audio_encoding(
        input_path, output_path,
        model_name, model_arch, model_weight
    )
    logging.info("Extracted audio feature to: {}".format(output_path))
    logging.info("Audio feature shape: {}".format(feature_shape))

    return feature_shape


def video_to_images(input_path: str,
                    output_dir: str,
                    fps: int = 25):
    """
    Params:
        input_path(str): 输入视频路径;
        output_dir(str): 输出图像文件夹路径;
        fps(int): 输出图像的帧率, 默认为25;
    """

    assert os.path.exists(input_path), \
        logging.error("The input video path does not exist.")

    cmd = "ffmpeg -loglevel error -i {} -vf fps={} -qmin 1 -q:v 1 -start_number 0 {}".format(
        input_path, fps, os.path.join(output_dir, "%d.jpg"))
    logging.info("[Commond]:\n{}".format(cmd))
    os.system(cmd)

    logging.info("Saved video frames to: {}".format(output_dir))


def run_face_parsing(input_dir: str,
                     output_dir: str,
                     save_label_dir: str,
                     device: str = "cpu",
                     gpu_id: int = -1
                     ) -> None:
    """
    Params:
        input_dir(str): 输入图像文件夹路径;
        output_dir(str): 输出图像文件夹路径;
        save_label_dir(str): 保存label图像文件夹路径;
        device(str): device;
        gpu_id(int): gpu index;

    """
    assert os.path.exists(input_dir), \
        logging.error("The input image dir does not exist.")

    bise_face_parser = BiseNet_FaceParser(
        model=face_seg_bisenet_weight,
        device=device, gpu_id=gpu_id
    )

    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    pbar = tqdm(total=len(image_paths))
    for image_path in image_paths:
        image_bgr = cv2.imdecode(np.fromfile(image_path, dtype="uint8"), 1)
        bise_label = bise_face_parser.parsing(image=image_bgr.copy())
        image_mask = bise_face_parser.get_head_mask(bise_label)

        save_name = "{}.png".format(
            os.path.splitext(os.path.basename(image_path))[0])
        save_mask_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_mask_path, image_mask)

        save_lable_path = os.path.join(save_label_dir, save_name)
        cv2.imwrite(save_lable_path, bise_label)

        pbar.update(1)

    pbar.close()

    logging.info("Saved face parse images to: {}".format(output_dir))
    logging.info("Saved face label images to: {}".format(save_label_dir))


def run_face_landmark(input_dir: str,
                      device: str = "cpu",
                      gpu_id: int = -1
                      ) -> None:
    """
    Params:
        input_dir(str): 输入图像文件夹路径;
        device(str): device;
        gpu_id(int): gpu index;
    """
    assert os.path.exists(input_dir), \
        logging.error("The input image dir does not exist.")

    estimator = FaceAlignment_Estimator(
        device=device, gpu_id=gpu_id
    )
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    pbar = tqdm(total=len(image_paths))
    for image_path in image_paths:
        _, _, landmarks = estimator.estimation(image_path)
        lmks = landmarks[0, :, :2].reshape(-1, 2)
        np.savetxt(image_path.replace("jpg", "lms"), lmks, "%f")
        pbar.update(1)
    pbar.close()

    logging.info("Saved the face landmarks data to: {}".format(input_dir))


def run_background_extraction(input_dir: str,
                              parse_dir: str,
                              skip_steps: int = 24,
                              ) -> None:
    # AD-NeRF Origin Solution
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_paths = [image_paths[i]
                   for i in range(0, len(image_paths), skip_steps)]
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    h, w = tmp_image.shape[:2]
    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path)
        parse_image_path = os.path.join(
            parse_dir, image_name.replace(".jpg", ".png"))
        parse_img = cv2.imread(parse_image_path, cv2.IMREAD_COLOR)
        bg = (parse_img[..., 0] == 255) & \
            (parse_img[..., 1] == 255) & \
            (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)

    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]

    imgs = []
    num_pixs = distss.shape[1]
    for image_path in tqdm(image_paths):
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)

    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = \
        bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]

    base_dir = os.path.dirname(input_dir)
    cv2.imwrite(os.path.join(base_dir, "bc.jpg"), bc_img)

    logging.info("Saved the background image to: {}".format(
        os.path.join(base_dir, "bc.jpg")))


def run_torso_and_gt_extraction(input_dir: str):
    # AD-NeRF Origin Solution
    from scipy.ndimage import binary_dilation
    base_dir = os.path.dirname(input_dir)

    # load bg
    bg_image = cv2.imread(os.path.join(
        base_dir, "bc.jpg"), cv2.IMREAD_UNCHANGED)
    # load ori images
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

    for image_path in tqdm(image_paths):
        # read ori image
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3]

        # read semantics
        seg = cv2.imread(image_path.replace(
            "ori_imgs", "parsing").replace(".jpg", ".png"))
        head_part = (seg[..., 0] == 255) & (
            seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_part = (seg[..., 0] == 0) & (
            seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_part = (seg[..., 0] == 0) & (
            seg[..., 1] == 0) & (seg[..., 2] == 255)
        bg_part = (seg[..., 0] == 255) & (
            seg[..., 1] == 255) & (seg[..., 2] == 255)

        # get gt image
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace("ori_imgs", "gt_imgs"), gt_image)

        # get torso image
        torso_image = gt_image.copy()  # rgb
        torso_image[head_part] = bg_image[head_part]
        torso_alpha = 255 * \
            np.ones((gt_image.shape[0], gt_image.shape[1],
                    1), dtype=np.uint8)  # alpha

        # torso part "vertical" in-painting...
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1)  # [M, 2]
        # lexsort: sort 2D coords first by y then by x,
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(
            torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid]  # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)]
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = gt_image[tuple(top_torso_coords.T)]  # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(
                L, 0)  # [L, m, 2]
            inpaint_offsets = np.stack(
                [-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]  # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            # [Lm, 2]
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2)
            inpaint_torso_colors = top_torso_colors[None].repeat(
                L, 0)  # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1)  # [L, 1, 1]
            inpaint_torso_colors = (
                inpaint_torso_colors * darken_scaler).reshape(-1, 3)  # [Lm, 3]
            # set color
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            inpaint_torso_mask = np.zeros_like(
                torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None

        # neck part "vertical" in-painting...
        push_down = 4
        L = 48 + push_down + 1

        neck_part = binary_dilation(neck_part, structure=np.array(
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), iterations=3)

        neck_coords = np.stack(np.nonzero(neck_part), axis=-1)  # [M, 2]
        # lexsort: sort 2D coords first by y then by x,
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(
            neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid]  # [m, 2]
        # only keep top-is-head pixels
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)]

        top_neck_coords = top_neck_coords[mask]
        # push these top down for 4 pixels to make the neck inpainting more natural...
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down,
                                    np.zeros_like(offset_down)], axis=-1)
        # get the color
        top_neck_colors = gt_image[tuple(top_neck_coords.T)]  # [m, 3]
        # construct inpaint coords (vertically up, or minus in x)
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0)  # [L, m, 2]
        inpaint_offsets = np.stack(
            [-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]  # [L, 1, 2]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2)  # [Lm, 2]
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0)  # [L, m, 3]
        darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1)  # [L, 1, 1]
        inpaint_neck_colors = (inpaint_neck_colors *
                               darken_scaler).reshape(-1, 3)  # [Lm, 3]
        # set color
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        # apply blurring to the inpaint area to avoid vertical-line artifects...
        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)

        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # set mask
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0

        cv2.imwrite(image_path.replace("ori_imgs", "torso_imgs").replace(
            ".jpg", ".png"), np.concatenate([torso_image, torso_alpha], axis=-1))

    logging.info("Saved the torso images to: {}".format(
        os.path.join(base_dir, "torso_imgs")))


def run_face_feature_extraction(input_path: str,
                                temp_dir: str,
                                output_dir: str):
    """
    Params:
        input_path(str): 输入视频路径;
        temp_dir(str): 输出"face_feature"数据保存目录路径;
        output_dir(str): 输出"au.csv"数据保存目录路径;
    """
    assert os.path.exists(input_path), \
        logging.error("The input video path does not exist.")

    cmd = "FeatureExtraction -f {} -out_dir {}".format(input_path, temp_dir)
    logging.info("[Commond]:\n{}".format(cmd))
    os.system(cmd)

    shutil.copy(os.path.join(temp_dir, "face_data.csv"),
                os.path.join(output_dir, "au.csv"))
    shutil.rmtree(temp_dir)

    logging.info("Save the face feature to: {}".format(
        os.path.join(output_dir, "au.csv")))


def run_face_tracking(input_dir: str,
                      ThreeDMM_dir: str = ThreeDMM_weight
                      ) -> None:
    """
    Params:
        input_dir(str): 模特的头部区域视频帧图像保存目录;
        ThreeDMM_dir(str): 3DMM模型保存目录;
    """
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    temp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    image_h, image_w = temp_image.shape[:2]
    face_tracker_training(
        ori_img_dir=input_dir,
        num_frames=len(image_paths),
        ThreeDMM_dir=ThreeDMM_dir,
        img_h=image_h, img_w=image_w
    )
    logging.info("Face Tracking Done.")


def save_transformers(input_dir: str) -> None:
    base_dir = os.path.dirname(input_dir)

    import torch

    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)  # [H, W, 3]
    h, w = tmp_image.shape[:2]

    params_dict = torch.load(os.path.join(base_dir, "track_params.pt"))
    focal_len = params_dict["focal"]
    euler_angle = params_dict["euler"]
    trans = params_dict["trans"] / 10.0
    valid_num = euler_angle.shape[0]

    def euler2rot(euler_angle):
        batch_size = euler_angle.shape[0]
        theta = euler_angle[:, 0].reshape(-1, 1, 1)
        phi = euler_angle[:, 1].reshape(-1, 1, 1)
        psi = euler_angle[:, 2].reshape(-1, 1, 1)
        one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                         device=euler_angle.device)
        zero = torch.zeros((batch_size, 1, 1),
                           dtype=torch.float32, device=euler_angle.device)
        rot_x = torch.cat((
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ), 2)
        rot_y = torch.cat((
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ), 2)
        rot_z = torch.cat((
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1)
        ), 2)
        return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

    # train_val_split = int(valid_num*0.5)
    # train_val_split = valid_num - 25 * 20 # take the last 20s as valid set.
    train_val_split = int(valid_num * 10 / 11)

    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)

    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))

    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ["train", "val"]
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    for split in range(2):
        transform_dict = dict()
        transform_dict["focal_len"] = float(focal_len[0])
        transform_dict["cx"] = float(w/2.0)
        transform_dict["cy"] = float(h/2.0)
        transform_dict["frames"] = []
        ids = train_val_ids[split]
        save_id = save_ids[split]

        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict["img_id"] = i
            frame_dict["aud_id"] = i

            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]

            frame_dict["transform_matrix"] = pose.numpy().tolist()

            transform_dict["frames"].append(frame_dict)

        with open(os.path.join(base_dir, "transforms_" + save_id + ".json"), "w") as fp:
            json.dump(transform_dict, fp, indent=2, separators=(",", ": "))

    logging.info("Transforms Saved.")


def preprocess_tasks(
    # general config
    input_video: str, data_dir: str,
    train_dir: str, infer_dir: str,
    save_model_cfg: str = None,
    save_log: str = None,
    device: str = "cpu",
    gpu_id: int = -1,
    task: Union[int, list] = -1,
    # audio/video config
    quality: int = 18,
    max_seconds: int = 600,
    target_fps: int = 25,
    target_sr: int = 16000,
    # checking config
    checker_only_first: bool = False,
    checker_skip_steps: int = 24,
    # matting config
    background_ref: Union[str, tuple, list, np.ndarray] = "green",
    background_trt: Union[str, tuple, list, np.ndarray] = "gray",
    # audio feature extraction config
    afe_model_name: str = "hubert",
    afe_model_arch: str = "large",
    afe_model_weight: str = "facebook/hubert-large-ls960-ft",
    # nerf structure
    head_region_size: int = 512,
    audio_net_ndim: int = 32,
    # train params
    train_head_iters: int = 200000,
    train_lips_iters: int = 250000,
    train_torso_iters: int = 200000,
    # infer params
    use_torso: bool = True,
    merge_mode: int = 0
) -> None:
    """
    Params:
        # =====>> general config
        input_video(str): 输入的模特视频路径;
        data_dir(str): 模特资源数据保存路径;
        train_dir(str): 模特数据训练权重保存路径;
        infer_dir(str): 模特推理结果视频保存路径;
        save_model_cfg(str): 保存模型配置文件, 如果为`None`, 则会保存到`data_dir`下;
        save_log(str): 保存模型数据前处理日志, 如果为`None`, 则不会保存;
        device(str): 使用设备, "cpu" or "cuda";
        gpu_id(int): 设备索引, 如果`device`为"cpu", 则为-1, 如果`device`为"cuda", 则指gpu索引;
        task(int, list): 待执行的任务索引, 默认-1, 执行所有任务;
        # =====>> audio/video config
        quality(int): 视频格式转换时的`-crf`质量参数, 默认18;
        max_seconds(int): 最长支持的视频时长(单位: 秒), 默认600;
        target_fps(int): 转换后视频的帧率(单位: 秒), 默认25;
        target_sr(int): 转换后音频的采样率, 默认16000;
        # =====>> checking config
        checker_only_first(bool): 输入视频审核时, 是否启用"仅首帧模型", 默认False;
        checker_skip_steps(int): 输入视频审核时, 抽取的每两帧之间的间隔, 默认24,
        # =====>> matting config
        background_ref(str): 参考背景, 默认"green";
        background_trt(str): 目标背景, 默认"gray";
        # =====>> audio feature extraction
        afe_model_name(str): 音频特征提取所使用的算法名称, 默认"hubert";
        afe_model_arch(str): 音频特征提取所使用的算法结构, "base" or "large", 默认"large";
        afe_model_weight(str): 音频特征提取所使用的算法权重, huggingface model name or local dir path,
            默认"facebook/hubert-large-ls960-ft";
        # =====>> nerf structure
        head_region_size(int): 头像部分视频尺寸, 默认512;
        audio_net_ndim(int): audio net的输入特征维度数, 默认32;
        # =====>> train params
        train_head_iters(int): 训练阶段, 头部训练迭代次数, 默认200000;
        train_lips_iters(int): 训练阶段, 嘴部微调迭代次数, 默认250000;
        train_torso_iters(int): 训练阶段, 躯干部训练迭代次数, 默认200000;
        # =====>> infer params
        use_torso(bool): 推理阶段, 是否使用包含驱赶部参数的权重, 默认True;
        merge_mode(int): 推理结果的后处理模型, 默认0;

    """
    logging.basicConfig(filename=save_log, level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    st_time = time.time()

    os.makedirs(data_dir, exist_ok=True)
    # 模特名称
    model_name = os.path.basename(data_dir)
    # 模特资源视频(临时文件)
    resource_temp_video_path = os.path.join(data_dir, "input_temp.mp4")
    # 模特资源视频
    resource_video_path = os.path.join(data_dir, "input.mp4")
    # 模特资源视频的抠图视频
    resource_alpha_path = os.path.join(data_dir, "alpha.mp4")
    # 模特资源视频首帧预览图
    preview_image_path = os.path.join(data_dir, "preview.png")
    # 模特的头部区域视频
    face_data_video_path = os.path.join(data_dir, "face_data.mp4")
    # 模特资源视频的音频
    resource_audio_path = os.path.join(data_dir, "audio.wav")
    # 模特资源视频的音频特征
    resource_audio_feat_path = os.path.join(data_dir, "audio_feat.npy")
    # 模特的头部区域视频帧图像
    ori_imgs_dir = os.path.join(data_dir, "ori_imgs")
    os.makedirs(ori_imgs_dir, exist_ok=True)
    # 模特的头部区域人脸解析结果
    parsing_imgs_dir = os.path.join(data_dir, "parsing")
    os.makedirs(parsing_imgs_dir, exist_ok=True)
    parsing_label_dir = os.path.join(data_dir, "parsing_label")
    os.makedirs(parsing_label_dir, exist_ok=True)
    # 模特的头部区域ground truth
    gt_imgs_dir = os.path.join(data_dir, "gt_imgs")
    os.makedirs(gt_imgs_dir, exist_ok=True)
    # 模特的头部区域torso images
    torso_imgs_dir = os.path.join(data_dir, "torso_imgs")
    os.makedirs(torso_imgs_dir, exist_ok=True)
    # 模特的头部区域openface feature extraction结果(临时目录)
    facefeature_temp_dir = os.path.join(data_dir, "facefeature")
    os.makedirs(facefeature_temp_dir, exist_ok=True)
    # 模特配置文件
    model_cfg_file = os.path.join(data_dir, "model_data.json")

    # ==========>> 前处理任务流程开始 <<==========
    if (task == -1) or (1 in task):
        logging.info("Task 01: Video Format Conversion")
        video_format_conversion(
            input_path=input_video,
            output_path=resource_temp_video_path,
            quality=quality,
            target_fps=target_fps,
            max_seconds=max_seconds
        )

    if (task == -1) or (2 in task):
        logging.info("Task 02: Video Matting")
        video_matting(
            input_path=resource_temp_video_path,
            background_ref=background_ref,
            output_alpha_path=resource_alpha_path,
            background_trt=background_trt,
            output_video_path=resource_video_path,
            device=device,
            gpu_id=gpu_id
        )

    if (task == -1) or (3 in task):
        logging.info("Task 03: Save Preview Image")
        save_preview_image(
            input_path=resource_video_path,
            output_path=preview_image_path
        )

    if (task == -1) or (4 in task):
        logging.info("Task 04: Head Region Detection and Crop & Resize")
        head_bbox = head_region_detection(
            input_path=resource_video_path,
            alpha_path=resource_alpha_path,
            output_path=face_data_video_path,
            only_first=checker_only_first,
            skip_steps=checker_skip_steps,
            quality=quality,
            target_size=head_region_size,
            device=device,
            gpu_id=gpu_id
        )

    if (task == -1) or (5 in task):
        logging.info("Task 05: Audio Stream Extraction")
        audio_stream_extraction(
            input_path=face_data_video_path,
            output_path=resource_audio_path,
            sample_rate=target_sr,
            mono=True
        )

    if (task == -1) or (6 in task):
        logging.info("Task 06: Audio Feature Extraction")
        audio_feat_shape = audio_feature_extraction(
            input_path=resource_audio_path,
            output_path=resource_audio_feat_path,
            model_name=afe_model_name,
            model_arch=afe_model_arch,
            model_weight=afe_model_weight,
            device=device,
            gpu_id=gpu_id
        )

    if (task == -1) or (7 in task):
        logging.info("Task 07: Video to Images")
        video_to_images(
            input_path=face_data_video_path,
            output_dir=ori_imgs_dir,
            fps=target_fps
        )

    if (task == -1) or (8 in task):
        logging.info("Task 08: Run Face Parsing")
        run_face_parsing(
            input_dir=ori_imgs_dir,
            output_dir=parsing_imgs_dir,
            save_label_dir=parsing_label_dir,
            device=device,
            gpu_id=gpu_id
        )

    if (task == -1) or (9 in task):
        logging.info("Task 09: Run Face Landmark")
        run_face_landmark(
            input_dir=ori_imgs_dir,
            device=device,
            gpu_id=gpu_id
        )

    if (task == -1) or (10 in task):
        logging.info("Task 10: Run Background Extraction")
        run_background_extraction(
            input_dir=ori_imgs_dir,
            parse_dir=parsing_imgs_dir,
            skip_steps=24,
        )

    if (task == -1) or (11 in task):
        logging.info("Task 11: Run Torso and Ground Truth Extraction")
        run_torso_and_gt_extraction(
            input_dir=ori_imgs_dir
        )

    if (task == -1) or (12 in task):
        logging.info("Task 12: Run OpenFace FeatureExtraction")
        run_face_feature_extraction(
            input_path=face_data_video_path,
            temp_dir=facefeature_temp_dir,
            output_dir=data_dir
        )

    if (task == -1) or (13 in task):
        logging.info("Task 13: Run Face Tracking")
        run_face_tracking(input_dir=ori_imgs_dir)

    if (task == -1) or (14 in task):
        logging.info("Task 14: Save `transforms.json`")
        save_transformers(input_dir=ori_imgs_dir)
    # ==========>> 前处理任务流程结束 <<==========

    # 保存模特配置文件
    if task == -1:
        logging.info("Process Done, Save Model Config.")
        video_reader = VideoReader(resource_video_path)
        model_config = {
            # general config
            "model_name": model_name,
            "data_dir": os.path.realpath(data_dir),
            "train_dir": os.path.basename(train_dir),
            "infer_dir": os.path.basename(infer_dir),
            # audio/video config
            "video_file": "input.mp4",
            "audio_file": "audio.wav",
            "alpha_file": "alpha.mp4",
            "frame_height": video_reader.shape[0],
            "frame_width": video_reader.shape[1],
            "frame_count": len(video_reader),
            "fps": target_fps,
            "sample_rate": target_sr,
            # matting
            "background_ref": background_ref,
            "background_trt": background_trt,
            # head region
            "head_video_file": "face_data.mp4",
            "head_region": head_bbox,
            "head_size": head_region_size,
            # audio feature extraction
            "alpha_feat_file": "audio_feat.npy",
            "afe_model_name": afe_model_name,
            "afe_model_arch": afe_model_arch,
            "afe_model_weight": afe_model_weight,
            # nerf structure
            "nerf_structure": {
                "audio_in_dim": int(audio_feat_shape[-1]),
                "audio_out_dim": int(audio_net_ndim),
                "ind_num": 10000 if len(video_reader) < video_reader.fps*60*5 else 20000,
            },
            # train params
            "train_params": {
                "train_head_iters": train_head_iters,
                "train_lips_iters": train_lips_iters,
                "train_torso_iters": train_torso_iters,
            },
            # infer params
            "infer_params": {
                # 如果为True, 使用躯干部的权重, 如果为False, 则会使用头部权重;
                "use_torso": use_torso,
                # 后处理模式, 不同的模式对应不同的后处理逻辑;
                "merge_mode": merge_mode,
                # 推理时所加载的数据起始帧和终止帧索引;
                "frame_index_start_at": 0,
                "frame_index_end_at": -1
            }
        }
        video_reader.close()

        with open(model_cfg_file, "w", encoding="utf-8") as f1:
            json.dump(model_config, f1)
        if save_model_cfg is not None:
            with open(save_model_cfg, "w", encoding="utf-8") as f2:
                json.dump(model_config, f2)

    ed_time = time.time()
    logging.info("Total time: {:.4f}ms".format((ed_time-st_time)*1e3))

    return save_model_cfg if save_model_cfg is not None else model_cfg_file
