import argparse

import torch

from ernerf.networks.network import NeRFNetwork
from ernerf.networks.provider import NeRFDataset
from ernerf.networks.utils import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def train_task(
    data_path: str,
    workspace: str,
    asr_model: str = "",
    data_range: list = [0, -1],
    #
    finetune_lips: bool = False,
    torso: bool = False,
    head_ckpt: str = None,
    #
    ind_num: int = 10000,
    audio_in_dim=-1,
    audio_out_dim=32,
    iters: int = 200000,
    patch_size: int = 1,
    #
    preload: int = 0,
):
    """ER-NeRF的训练任务函数

    Params:
        data_path(str): 模特数据目录路径;
        workspace(str): 训练日志及模型保存目录路径;
        asr_model(str): 使用的音频特征提取模型路径(或是 huggingface model name);
            only support hubert for now;
        data_range(list): 模特数据帧数抽取索引;
        finetune_lips(bool): 嘴部的微调训练, 在完成头部训练后进行;
        torso(bool): 躯干部的训练, 在完成头部训练和嘴部微调后进行;
        head_ckpt(str): 已经训练好的头部模型文件路径, 当`torso`为`True`时生效;
        ind_num(int): number of individual codes, should be larger than training dataset size;
        iters(int): training iters, epochs = iters / train_data_length;
        patch_size(int): [experimental] render patches in training, so as to apply LPIPS loss;
            1 means disabled, use [64, 32, 16] to enable;
        preload(int): 数据预加载方式类型;
            0 - 不进行预加载, 实时从硬盘加载数据;
            1 - 预加载数据到内存, 可以较快加速训练流程, 但增加内存消耗;
            2 - 预加载数据到显存, 可以更快加速训练流程, 但增加显存消耗;
    """

    # ==========>> Initial Training Params <<==========
    opt = argparse.Namespace()
    #
    opt.path: str = data_path
    opt.O: bool = True
    opt.test: bool = False
    opt.test_train: bool = False
    opt.data_range: list = data_range
    opt.workspace: str = workspace
    opt.asr_model: str = asr_model
    #
    opt.finetune_lips: bool = finetune_lips
    opt.torso: bool = torso
    opt.head_ckpt: str = head_ckpt
    #
    opt.ind_num: int = ind_num
    opt.iters: int = iters
    opt.patch_size: int = patch_size
    #
    opt.preload: int = preload
    #
    opt.seed: int = 0
    opt.fp16: bool = True
    opt.lr: float = 1e-2
    opt.lr_net: float = 1e-3
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
    opt.aud: str = ""
    opt.emb: bool = False
    opt.ind_dim: int = 4
    opt.ind_dim_torso: int = 8
    opt.amb_dim: int = 2
    opt.part: bool = False
    opt.part2: bool = False
    opt.train_camera: bool = False
    opt.smooth_path: bool = False
    opt.smooth_path_window: int = 7
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
    os.makedirs(opt.workspace, exist_ok=True)

    seed_everything(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeRFNetwork(opt, audio_in_dim=audio_in_dim,
                        audio_out_dim=audio_out_dim)

    # manually load state dict for head
    if opt.torso and opt.head_ckpt != "":
        model_dict = torch.load(opt.head_ckpt, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(
            model_dict, strict=False)

        if len(missing_keys) > 0:
            print(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[WARN] unexpected keys: {unexpected_keys}")

        # freeze these keys
        for k, v in model.named_parameters():
            if k in model_dict:
                print(f"[INFO] freeze {k}, {v.shape}")
                v.requires_grad = False

    criterion = torch.nn.MSELoss(reduction="none")

    def optimizer(model): return torch.optim.AdamW(
        model.get_params(opt.lr, opt.lr_net), betas=(0, 0.99), eps=1e-8)

    train_loader = NeRFDataset(
        opt, device=device, type="train").dataloader()

    assert len(train_loader) < opt.ind_num, \
        f"[ERROR] dataset too many frames: {len(train_loader)}, please increase --ind_num to this number!"

    # temp fix: for update_extra_states
    model.aud_features = train_loader._data.auds
    model.eye_area = train_loader._data.eye_area
    model.poses = train_loader._data.poses

    # decay to 0.1 * init_lr at last iter step
    if opt.finetune_lips:
        def scheduler(optimizer): return optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.05 ** (iter / opt.iters))
    else:
        def scheduler(optimizer): return optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.5 ** (iter / opt.iters))

    metrics = [PSNRMeter(), LPIPSMeter(device=device)]

    eval_interval = max(1, int(5000 / len(train_loader)))
    trainer = Trainer(
        "ngp", opt, model, device=device, workspace=opt.workspace,
        optimizer=optimizer, criterion=criterion,
        ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
        scheduler_update_every_step=True, metrics=metrics,
        use_checkpoint=opt.ckpt, eval_interval=eval_interval
    )

    with open(os.path.join(opt.workspace, 'opt.txt'), 'a') as f:
        f.write(str(opt))

    valid_loader = NeRFDataset(
        opt, device=device, type="val", downscale=1).dataloader()

    max_epochs = np.ceil(
        opt.iters / len(train_loader)).astype(np.int32)
    print(f"[INFO] max_epoch = {max_epochs}")
    trainer.train(train_loader, valid_loader, max_epochs)

    # free memory
    del train_loader, valid_loader
    torch.cuda.empty_cache()

    # also test
    test_loader = NeRFDataset(
        opt, device=device, type='test').dataloader()

    if test_loader.has_gt:
        # blender has gt, so evaluate it.
        trainer.evaluate(test_loader)

    trainer.test(test_loader)

    return


if __name__ == "__main__":
    pass
