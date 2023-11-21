# Avatar ERNeRF

## ER-NeRF : [Paper](https://arxiv.org/abs/2307.09323) | [github](https://github.com/Fictionarry/ER-NeRF.git)

## 语言: [[English](README.md)] | [简体中文]

## 概述

- 这只是一个缝合怪项目，一个基于ER-NeRF算法的数字人视频生成的全流程方案，提供额外的预处理和后处理。本项目的作者并不是ER-NeRF算法的作者，没有足够的能力回答通过ER-NeRF算法训练后得到的结果不足够好的问题。

## 安装

- Ubuntu18.04; CUDA11.3; CUDNN>=8.2.4, <8.7.0; gcc/g++-9;

- 第三方库:

    ```shell
    sudo apt-get install libasound2-dev portaudio19-dev # dependency for pyaudio
    sudo apt-get install ffmpeg # or build from source
    # build openface from source `https://github.com/TadasBaltrusaitis/OpenFace.git`
    ```

- python环境:

    ```shell
    conda create -n ernerf python=3.10 -y
    conda activate ernerf

    # install pytorch
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

    # install pytorch3d
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"

    # install others
    pip install -r requirements.txt

    sh install_ext.sh
    ```

- 预训练模型:

    Download the pretrained weights from: [Google Drive](https://drive.google.com/file/d/12kz5-UwWyKzTf7z2hFUO41Jx5wnTEbJy/view?usp=drive_link)

    Download 3DMM model for head pose estimation:
    ```shell
    wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/exp_info.npy?raw=true -O pretrained_weights/3DMM/exp_info.npy
    wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/keys_info.npy?raw=true -O pretrained_weights/3DMM/keys_info.npy
    wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/sub_mesh.obj?raw=true -O pretrained_weights/3DMM/sub_mesh.obj
    wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/topology_info.npy?raw=true -O pretrained_weights/3DMM/topology_info.npy
    ``` 

    Download 3DMM model from [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details):
    ```shell
    # 1. copy 01_MorphableModel.mat to `pretrained_weights/3DMM`
    # 2.
    cd modules/face_tracking
    python convert_BFM.py
    ```

## 使用说明

- 输入的视频数据应为5-10分钟左右的视频, 视频中只有一人, 且需要保证时间连续性(一镜到底).

    ```shell
    cd <PROJECT>
    export PYTHONPATH=./

    python -u create_train_task.py -i <input_video> --model_uid <model_name>

    python -u create_infer_task.py -i <input_audio> -c <model_name or model_config_file>
    ```

## 感谢列表

- 人脸检测模型来源于: [yolov7-face](https://github.com/derronqi/yolov7-face.git)

- 人脸特征提取来源于: [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace.git)

- 人脸关键点检测模型来源于: [face-alignment](https://github.com/1adrianb/face-alignment.git)

- 人脸解析模型来源于: [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch.git)

- 人脸跟踪模型来源于: [AD-NeRF](https://github.com/YudongGuo/AD-NeRF.git)

- 人体姿态估计模型来源于: [yolov7-pose](https://github.com/trancongman276/yolov7-pose.git)

- 背景抠图模型来源于: [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2.git)

- 说话头生成模型来源于: [ER-NeRF](https://github.com/Fictionarry/ER-NeRF.git)

## 其他

- 一个微信的技术分享群, 欢迎分享和交流
![wechat](./docs/wechat_group.jpg)
