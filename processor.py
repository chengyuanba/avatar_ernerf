"""
数据前处理流程
"""

import json
import logging
import os
import time
from typing import Union

import numpy as np

from tasks.preprocess import preprocess_tasks


def processor(
    video_path: str, model_uid: str,
    train_dir: str = None,
    infer_dir: str = None,
    save_model_cfg: str = None,
    device: str = "cuda",
    gpu_id: int = 0,
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
) -> str:

    ROOT = os.path.dirname(os.path.abspath(__file__))

    # ==================== 初始化日志配置 ====================
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # ==================== 解析全局配置 ====================
    global_config = "{}/global_config.json".format(ROOT)
    if not os.path.exists(global_config):
        raise FileNotFoundError(
            f"Not Found global config file: {global_config}")
    with open(global_config, "r", encoding="utf-8") as f:
        global_config = json.load(f)
    # 获取全局参数:
    model_dataset_folder = global_config["model_dataset_folder"]
    # 存放该模特的视频经过前处理后得到的资源数据
    data_dir = os.path.join(model_dataset_folder, model_uid)
    os.makedirs(data_dir, exist_ok=True)

    # ==================== 检查输入视频 ====================
    if os.path.isfile(video_path):
        temp_video_path = video_path
    else:
        raise FileNotFoundError("Not Found video file: {}".format(video_path))

    # ==================== 视频预处理 ====================
    process_time = time.time()
    model_cfg_file = preprocess_tasks(
        # general config
        input_video=temp_video_path,
        data_dir=data_dir,
        train_dir=train_dir,
        infer_dir=infer_dir,
        save_model_cfg=save_model_cfg,
        device=device,
        gpu_id=gpu_id,
        task=task,
        # audio/video config
        quality=quality,
        max_seconds=max_seconds,
        target_fps=target_fps,
        target_sr=target_sr,
        # checking config
        checker_only_first=checker_only_first,
        checker_skip_steps=checker_skip_steps,
        # matting config
        background_ref=background_ref,
        background_trt=background_trt,
        # audio feature extraction config
        afe_model_name=afe_model_name,
        afe_model_arch=afe_model_arch,
        afe_model_weight=afe_model_weight,
        # nerf structure
        head_region_size=head_region_size,
        audio_net_ndim=audio_net_ndim,
        # train params
        train_head_iters=train_head_iters,
        train_lips_iters=train_lips_iters,
        train_torso_iters=train_torso_iters,
        # infer params
        use_torso=use_torso,
        merge_mode=merge_mode
    )
    process_time = time.time() - process_time
    logging.info("PreProcess Time: {:.4f}s".format(process_time))

    return model_cfg_file
