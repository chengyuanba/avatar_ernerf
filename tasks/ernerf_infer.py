import argparse

import torch

from ernerf.networks.network import NeRFNetwork
from ernerf.networks.provider import NeRFDataset_Test
from ernerf.networks.utils import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def infer_task(
    data_path: str,
    workspace: str,
    driving_audio: str,
    data_range: list = [0, -1],
    #
    afe_model_weight: str = "facebook/hubert-large-ls960-ft",
    #
    ind_num: int = 10000,
    audio_in_dim: int = -1,
    audio_out_dim: int = 32,
    #
    preload: int = 0,
    add_torso: bool = True
):
    """
    Params:
        data_path(str): 模特资源数据目录;
        workspace(str): 训练好的权重目录;
        driving_audio(str): 驱动音频的特征数据文件路径;
        data_range(str): 加载数据帧的起始和终止索引;
        afe_model_weight(str): 音频特征提取所使用的算法权重, huggingface model name or local dir path,
            默认"facebook/hubert-large-ls960-ft";
        ind_num(int): 10000;
        audio_in_dim(int): -1;
        audio_out_dim(int): 32;
        preload(int): 0;
        add_torso(bool): True;
    """

    # ==========>> Initial Training Params <<==========
    opt = argparse.Namespace()
    #
    opt.path: str = data_path
    opt.O: bool = True
    opt.test_train: bool = True
    opt.data_range: list = data_range
    opt.workspace: str = workspace
    opt.asr_model: str = afe_model_weight
    opt.aud: str = driving_audio
    #
    opt.finetune_lips: bool = False
    opt.torso: bool = add_torso
    opt.head_ckpt: str = ""
    #
    opt.ind_num: int = ind_num
    opt.patch_size: int = 1
    #
    opt.preload: int = preload
    #
    opt.seed: int = 0
    opt.fp16: bool = True
    opt.lr: float = 1e-2
    opt.lr_net: float = 1e-2
    opt.ckpt: str = "latest"
    opt.num_rays: int = 4096 * 16
    opt.cuda_ray: bool = True
    opt.max_steps: int = 16
    opt.num_steps: int = 16
    opt.upsample_steps: int = 0
    opt.update_extra_interval: int = 16
    opt.max_ray_batch: int = 4096
    opt.warmup_step: int = 10000
    opt.amb_aud_loss: int = 1
    opt.amb_eye_loss: int = 1
    opt.unc_loss: int = 1
    opt.lambda_amb: float = 1e-4
    opt.bg_img: str = ""
    opt.fbg: bool = False
    opt.exp_eye: bool = True
    opt.fix_eye: float = -1.
    opt.smooth_eye: bool = False
    opt.torso_shrink: float = 0.8
    opt.color_space: str = "srgb"
    opt.bound: float = 1
    opt.scale: float = 4
    opt.offset: list = [0, 0, 0]
    opt.dt_gamma: float = 1/256
    opt.min_near: float = 0.05
    opt.density_thresh: float = 10
    opt.density_thresh_torso: float = 0.01
    opt.init_lips: bool = False
    opt.smooth_lips: bool = False
    opt.gui: bool = False
    opt.W: int = 450
    opt.H: int = 450
    opt.radius: float = 3.35
    opt.fovy: float = 21.24
    opt.max_spp: int = 1
    opt.att: int = 2
    opt.emb: bool = False
    opt.ind_dim: int = 4
    opt.ind_dim_torso: int = 8
    opt.amb_dim: int = 2
    opt.part: bool = False
    opt.part2: bool = False
    opt.train_camera: bool = False
    opt.smooth_path: bool = False
    opt.asr: bool = False
    opt.asr_wav: str = ""
    opt.asr_play: bool = False
    opt.asr_save_feats: bool = False
    opt.fps: int = 50
    opt.l: int = 10
    opt.m: int = 50
    opt.r: int = 10

    if opt.O:
        opt.fp16 = True
        opt.exp_eye = True

    assert opt.cuda_ray, "Only support CUDA ray mode."

    if opt.patch_size > 1:
        assert opt.num_rays % (opt.patch_size ** 2) == 0, \
            "patch_size ** 2 should be dividable by num_rays."
    # ==========>> Initial Training Params <<==========
    print(opt)

    seed_everything(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeRFNetwork(opt, audio_in_dim=audio_in_dim,
                        audio_out_dim=audio_out_dim)
    # init trainer
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace,
                      fp16=opt.fp16, use_checkpoint=opt.ckpt)

    # init data loader
    data_loader = NeRFDataset_Test(opt, device=device).dataloader()

    # temp fix: for update_extra_states
    model.aud_features = data_loader._data.auds
    model.eye_areas = data_loader._data.eye_area

    frame_rgb_list, frame_depth_list = trainer.test_new(data_loader)

    # free memory
    del data_loader
    torch.cuda.empty_cache()

    return frame_rgb_list, frame_depth_list


if __name__ == "__main__":
    pass
