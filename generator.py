"""
ER-NeRF推理+视频生成流程
"""

import json
import logging
import os
import time
import uuid

from tasks.ernerf_infer import infer_task
from tasks.postprocess import MergeProcessor
from tasks.preprocess import audio_feature_extraction


def generator(
    audio_path: str,
    model_cfg_file: str,
    second_start: float = 0,
    second_end: float = -1,
    background_rgb: tuple = (0, 0, 0),
    with_alpha: bool = True
) -> str:
    """
    Params:
        audio_path(str): 输入音频文件;
        model_cfg_file(str): 模特配置文件 or 模特唯一索引;
        second_start(float): 推理时, 使用资源数据的起始时间(单位: 秒);
        second_end(float): 推理时, 使用资源数据的终止时间(单位: 秒);
        background_rgb(tuple): 目标背景颜色(R, G, B), 默认黑色(0, 0, 0);
        with_alpha(bool): 输出结果是否包含alpha通道, 默认True;
    """

    ROOT = os.path.dirname(os.path.abspath(__file__))

    # ==================== 初始化日志配置 ====================
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # ==================== 解析全局配置 ====================
    global_config = "{}/global_config.json".format(ROOT)
    if not os.path.exists(global_config):
        raise FileNotFoundError(
            f"Not Found global config file: {global_config}")
    with open(global_config, "r", encoding="utf-8") as f1:
        global_config = json.load(f1)
    # 获取全局参数:
    model_dataset_folder = global_config["model_dataset_folder"]
    checkpoints_folder = global_config["checkpoints_folder"]
    results_folder = global_config["results_folder"]

    # ==================== 解析模特配置 ====================
    if not os.path.isfile(model_cfg_file):
        model_cfg_file = os.path.join(
            model_dataset_folder, model_cfg_file, "model_data.json")
    if not os.path.exists(model_cfg_file):
        raise FileNotFoundError(
            f"Not Found model config file: {model_cfg_file}")
    with open(model_cfg_file, "r", encoding="utf-8") as f2:
        model_config = json.load(f2)

    # 存放该模特的视频经过前处理后得到的资源数据
    data_dir = model_config["data_dir"]
    # 存放该模特的训练中间结果和权重文件
    ckpt_dir = os.path.join(
        checkpoints_folder, model_config["train_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)
    # #### 头部训练权重
    ckpt_dir_head = os.path.join(ckpt_dir, "head")
    os.makedirs(ckpt_dir_head, exist_ok=True)
    # #### 躯干部训练权重
    ckpt_dir_torso = os.path.join(ckpt_dir, "torso")
    os.makedirs(ckpt_dir_torso, exist_ok=True)
    # 存放该模特的推理结果
    save_dir = os.path.join(
        results_folder, model_config["infer_dir"])
    os.makedirs(save_dir, exist_ok=True)

    # ==================== 下载音频文件 ====================
    if os.path.isfile(audio_path):
        temp_audio_path = audio_path
    else:
        raise FileNotFoundError("Not Found audio file: {}".format(audio_path))

    # ================ Audio Feature Extraction ================
    audio_proc_time = time.time()
    # #### 音频特征提取结果与音频文件存储在同一目录
    temp_audio_feat_path = os.path.join(
        os.path.dirname(temp_audio_path),
        os.path.splitext(os.path.basename(temp_audio_path))[0] + "_feat.npy"
    )
    audio_feature_shape = audio_feature_extraction(
        input_path=temp_audio_path,
        output_path=temp_audio_feat_path,
        model_name=model_config["afe_model_name"],
        model_arch=model_config["afe_model_arch"],
        model_weight=model_config["afe_model_weight"],
        device="cuda", gpu_id=0
    )
    audio_proc_time = time.time() - audio_proc_time
    logging.info("Audio FeatureExtraction Time: {:.4f}ms".format(
        audio_proc_time*1e3))

    # ==================== ER-NeRF Inference ====================
    inference_time = time.time()
    # 起始帧索引
    frame_index_st = int(second_start * model_config["fps"])
    frame_index_st += model_config["infer_params"]["frame_index_start_at"]
    # 终止帧索引
    frame_index_ed = frame_index_st + audio_feature_shape[0] + 1
    if second_end == -1:
        if frame_index_ed >= model_config["frame_count"]:
            frame_index_ed = model_config["frame_count"] - 1
    else:
        if model_config["infer_params"]["frame_index_end_at"] == -1:
            if frame_index_ed >= model_config["frame_count"]:
                frame_index_ed = model_config["infer_params"]["frame_index_end_at"] - 1
        else:
            if frame_index_ed >= model_config["infer_params"]["frame_index_end_at"]:
                frame_index_ed = model_config["infer_params"]["frame_index_end_at"] - 1

    logging.info("Inference Start ...")
    frame_rgb_list, frame_depth_list = infer_task(
        data_path=data_dir,
        workspace=ckpt_dir_torso if model_config["infer_params"]["use_torso"] else ckpt_dir_head,
        driving_audio=temp_audio_feat_path,
        data_range=[frame_index_st, frame_index_ed],
        afe_model_weight=model_config["afe_model_weight"],
        ind_num=model_config["nerf_structure"]["ind_num"],
        audio_in_dim=model_config["nerf_structure"]["audio_in_dim"],
        audio_out_dim=model_config["nerf_structure"]["audio_out_dim"],
        add_torso=True
    )
    logging.info("Inference Done.")

    # ======================== PostProcess ========================
    merge_result_name = uuid.uuid1()
    if with_alpha:
        merge_result_path = os.path.join(save_dir, f"{merge_result_name}.webm")
    else:
        merge_result_path = os.path.join(save_dir, f"{merge_result_name}.mp4")

    logging.info("Merge Progress Start ...")
    merge_processor = MergeProcessor()
    merge_processor.run(
        model_config=model_config,
        audio_path=temp_audio_path,
        save_path=merge_result_path,
        head_frame_rgb_list=frame_rgb_list,
        head_frame_depth_list=frame_depth_list,
        frame_index_start=frame_index_st,
        frame_index_end=frame_index_ed,
        background_rgb=background_rgb,
        with_alpha=with_alpha,
        mode=model_config["infer_params"]["merge_mode"]
    )
    logging.info("Merge Head, Merge Audio Done.")

    inference_time = time.time() - inference_time
    logging.info("Inference Time total: {:.4f}ms; Per frame: {:.4f}ms.".format(
        inference_time*1e3, inference_time*1e3 / audio_feature_shape[0]))
    logging.info(f"Save result at: {merge_result_path}")

    return "Inference Done."
