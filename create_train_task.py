import argparse

from processor import processor
from trainer import trainer


def train(
    video_path: str,
    model_uid: str,
    train_dir: str = None,
    infer_dir: str = None,
    model_cfg_file: str = None,
    device: str = "cuda",
    gpu_id: int = 0,
    # checking config
    checker_only_first: bool = False,
    checker_skip_steps: int = 24,
    # matting config
    # background_ref: str = "green",
    background_ref: str = (105, 173, 107),
    background_trt: str = "gray",
    # audio feature extraction config
    afe_model_name: str = "hubert",
    afe_model_arch: str = "large",
    afe_model_weight: str = "facebook/hubert-large-ls960-ft",
    # nerf structure
    audio_net_ndim: int = 32,
    # train params
    train_head_iters: int = 200000,
    train_lips_iters: int = 250000,
    train_torso_iters: int = 200000,
    # infer params
    use_torso: bool = True,
    merge_mode: int = 0,
    # others
    preproc_only: bool = False,
    preproc_done: bool = False,
    preload: int = 0
) -> None:

    # ==================== Data Process ====================
    if not preproc_done:
        processor(
            video_path=video_path,
            model_uid=model_uid,
            train_dir=train_dir,
            infer_dir=infer_dir,
            save_model_cfg=model_cfg_file,
            device=device,
            gpu_id=gpu_id,
            task=-1,
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
            audio_net_ndim=audio_net_ndim,
            # train params
            train_head_iters=train_head_iters,
            train_lips_iters=train_lips_iters,
            train_torso_iters=train_torso_iters,
            # infer params
            use_torso=use_torso,
            merge_mode=merge_mode,
        )

    if preproc_only:
        return "Progress Done."

    # ==================== ERNeRF Train ====================
    trainer(
        model_cfg_file=model_uid if model_cfg_file is None else model_cfg_file,
        task_num=-1, preload=preload
    )

    return "Progress Done."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("--model_uid", type=str)
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--infer_dir", type=str)
    parser.add_argument("--model_cfg_file", type=str)
    args = parser.parse_args()

    train(
        video_path=args.input,
        model_uid=args.model_uid,
        train_dir=args.train_dir,
        infer_dir=args.infer_dir,
        model_cfg_file=args.model_cfg_file,
    )
