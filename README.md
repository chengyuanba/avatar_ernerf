# Avatar ERNeRF

## ER-NeRF : [Paper](https://arxiv.org/abs/2307.09323) | [github](https://github.com/Fictionarry/ER-NeRF.git)

## Overview

- This is just a suturing monster project, a whole process scheme of avatar video generation based on ER-NeRF algorithm to provide additional pre-processing and post-processing. The author is not the author of ER-NeRF algorithm, and does not have the ability to answer any problem with poor processing effect.

## Installation

- Ubuntu18.04; CUDA11.3; CUDNN>=8.2.4, <8.7.0; gcc/g++-9;

- third party

    ```shell
    sudo apt-get install libasound2-dev portaudio19-dev # dependency for pyaudio
    sudo apt-get install ffmpeg # or build from source
    # build openface from source `https://github.com/TadasBaltrusaitis/OpenFace.git`
    ```

- python env:

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

- pretrained_weights

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



## Usage

- The input video data should be a video of about 5-10 minutes, there is only one person in the video, and it needs to ensure time continuity (one shot to the end).

    ```shell
    cd <PROJECT>
    export PYTHONPATH=./

    python -u create_train_task.py -i <input_video> -n <model_name>

    python -u create_infer_task.py -i <input_audio> -c <model_name or model_config_file>
    ```

## Acknowledgement

- Face Detection From: [yolov7-face](https://github.com/derronqi/yolov7-face.git)

- Face FeatureExtraction From: [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace.git)

- Face Landmark Detection From: [face-alignment](https://github.com/1adrianb/face-alignment.git)

- Face Parsing From: [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch.git)

- Face Tracking From: [AD-NeRF](https://github.com/YudongGuo/AD-NeRF.git)

- Pose Estimation From: [yolov7-pose](https://github.com/trancongman276/yolov7-pose.git)

- Background Matting From: [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2.git)

- Talking Head From: [ER-NeRF](https://github.com/Fictionarry/ER-NeRF.git)
