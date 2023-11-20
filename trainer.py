"""
ER-NeRF训练流程
"""

import json
import logging
import os
import time
from typing import Union

from tasks.ernerf_train import train_task


def trainer(
    model_cfg_file: str,
    preload: int = 0,
    task: Union[int, list] = -1
) -> str:
    """
    Params:
        model_cfg_file(str): 模特配置文件 or 模特唯一索引;
        preload(int): 数据预加载方式类型;
            0 - 不进行预加载, 实时从硬盘加载数据;
            1 - 预加载数据到内存, 可以较快加速训练流程, 但增加内存消耗;
            2 - 预加载数据到显存, 可以更快加速训练流程, 但增加显存消耗;
        task(int, list): 训练任务类型;
            -1 - 进行所有训练任务;
            1 - 仅进行头部的训练任务;
            2 - 仅进行嘴部的微调训练任务;
            3 - 仅进行躯干部的训练任务;

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

    # ==================== ER-NeRF Training ====================
    training_time = time.time()
    # #### 模型训练: task head training
    if (task == -1) or (1 in task):
        logging.info("Training Step 01 [Head] Start ...")
        train_task(
            data_path=data_dir,
            workspace=ckpt_dir_head,
            asr_model=model_config["afe_model_weight"],
            data_range=[0, -1],
            ind_num=model_config["nerf_structure"]["ind_num"],
            audio_in_dim=model_config["nerf_structure"]["audio_in_dim"],
            audio_out_dim=model_config["nerf_structure"]["audio_out_dim"],
            iters=model_config["train_params"]["train_head_iters"],
            preload=preload,
            patch_size=1
        )
        logging.info("Training Step 01 [Head] Done.")

    # #### 模型训练: task lips fine-tune
    if (task == -1) or (2 in task):
        logging.info("Training Step 02 [Lips] Start ...")
        train_task(
            data_path=data_dir,
            workspace=ckpt_dir_head,
            asr_model=model_config["afe_model_weight"],
            data_range=[0, -1],
            ind_num=model_config["nerf_structure"]["ind_num"],
            audio_in_dim=model_config["nerf_structure"]["audio_in_dim"],
            audio_out_dim=model_config["nerf_structure"]["audio_out_dim"],
            iters=model_config["train_params"]["train_lips_iters"],
            preload=preload,
            patch_size=32,
            finetune_lips=True
        )
        logging.info("Training Step 02 [Lips] Done.")

    # #### 模型训练: step 03 -> torso training
    if (task == -1) or (3 in task):
        logging.info("Training Step 03 [Torso] Start ...")
        train_task(
            data_path=data_dir,
            workspace=ckpt_dir_torso,
            asr_model=model_config["afe_model_weight"],
            data_range=[0, -1],
            ind_num=model_config["nerf_structure"]["ind_num"],
            audio_in_dim=model_config["nerf_structure"]["audio_in_dim"],
            audio_out_dim=model_config["nerf_structure"]["audio_out_dim"],
            iters=model_config["train_params"]["train_torso_iters"],
            preload=preload,
            patch_size=1,
            torso=True,
            head_ckpt=os.path.join(ckpt_dir_head, "checkpoints", "ngp.pth")
        )
        logging.info("Training Step 03 [Torso] Done.")

    training_time = time.time() - training_time
    logging.info("Training Time: {:.4f}s".format(training_time))

    return "Training Done."
