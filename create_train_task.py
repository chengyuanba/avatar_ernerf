import argparse

from processor import processor
from trainer import trainer


def parse_args():
    parser = argparse.ArgumentParser()
    # ==========>> General
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input video path.")
    parser.add_argument("--model_uid", type=str, required=True,
                        help="Model unique identification.")
    parser.add_argument("--train_dir", type=str, required=False,
                        help="Trained checkpoint save directory.",
                        default=None)
    parser.add_argument("--infer_dir", type=str, required=False,
                        help="Inference result save directory.",
                        default=None)
    parser.add_argument("--model_cfg_file", type=str, required=False,
                        help="Model config file.",
                        default=None)
    parser.add_argument("--device", type=str, required=False,
                        choices=["cpu", "cuda"], help="Device to use.",
                        default="cuda")
    parser.add_argument("--gpu_id", type=int, required=False,
                        help="GPU ID to use.",
                        default=0)
    # ==========>> Checking
    parser.add_argument("--checker_only_first", type=bool, required=False,
                        help="Only check the first frame.",
                        default=False)
    parser.add_argument("--checker_skip_steps", type=int, required=False,
                        help="Skip steps for checking, if checker_only_first is True, it will be ignored.",
                        default=24)
    # ==========>> Matting
    parser.add_argument("--background_ref", type=str, required=False,
                        help="Background reference, can be a string means "
                        "color name: 'blue', 'green', ..."
                        "rgb value: '(105, 173, 107)', ... "
                        "file path: reference background image path.",
                        default="green")
    parser.add_argument("--background_trt", type=str, required=False,
                        help="Background target, can be a string means "
                        "color name: 'red', 'green', ..."
                        "rgb value: '(105, 173, 107)', ... "
                        "file path: target background image path.",
                        default="gray")
    # ==========>> Audio Feature
    parser.add_argument("--afe_model_name", type=str, required=False,
                        help="Audio feature extraction model name.",
                        default="hubert")
    parser.add_argument("--afe_model_arch", type=str, required=False,
                        help="Audio feature extraction model architecture.",
                        default="large")
    parser.add_argument("--afe_model_weight", type=str, required=False,
                        help="Audio feature extraction model weight. can be huggingface model or a file path.",
                        default="facebook/hubert-large-ls960-ft")
    # ==========>> NeRF
    parser.add_argument("--audio_net_ndim", type=int, required=False,
                        help="AudioNet ouptut feature's dimension.",
                        default=32)
    # ==========>> Training
    parser.add_argument("--train_head_iters", type=int, required=False,
                        help="Training head iterations.",
                        default=200000)
    parser.add_argument("--train_lips_iters", type=int, required=False,
                        help="Training lips iterations.",
                        default=250000)
    parser.add_argument("--train_torso_iters", type=int, required=False,
                        help="Training torso iterations.",
                        default=200000)
    # ==========>> Inference
    parser.add_argument("--use_torso", action="store_true",
                        help="Use torso or not when infer.")
    parser.add_argument("--merge_mode", type=int, default=0,
                        help="Merge mode when infer.")
    # ==========>> Others
    parser.add_argument("--preproc_only",  action="store_true",
                        help="Only do video data preprocess.")
    parser.add_argument("--preproc_done", action="store_true",
                        help="Preprocess has been done.")
    parser.add_argument("--preload", type=int, required=False, default=0,
                        help="Preload mode for train, 0 means from hard disk; 1 means from mermory; 2 means from gpu memory.")

    parser.set_defaults(use_torso=False)
    parser.set_defaults(preproc_only=False)
    parser.set_defaults(preproc_done=False)

    args = parser.parse_args()
    return args


def train(args) -> None:

    # ==================== Data Process ====================
    if not args.preproc_done:
        processor(
            video_path=args.video_path,
            model_uid=args.model_uid,
            train_dir=args.train_dir,
            infer_dir=args.infer_dir,
            save_model_cfg=args.model_cfg_file,
            device=args.device,
            gpu_id=args.gpu_id,
            task=-1,
            checker_only_first=args.checker_only_first,
            checker_skip_steps=args.checker_skip_steps,
            background_ref=args.background_ref,
            background_trt=args.background_trt,
            afe_model_name=args.afe_model_name,
            afe_model_arch=args.afe_model_arch,
            afe_model_weight=args.afe_model_weight,
            audio_net_ndim=args.audio_net_ndim,
            train_head_iters=args.train_head_iters,
            train_lips_iters=args.train_lips_iters,
            train_torso_iters=args.train_torso_iters,
            use_torso=args.use_torso,
            merge_mode=args.merge_mode,
        )

    if args.preproc_only:
        return "Progress Done."

    # ==================== ERNeRF Train ====================
    trainer(
        model_cfg_file=args.model_uid if args.model_cfg_file is None else args.model_cfg_file,
        task=-1, preload=args.preload
    )

    return "Progress Done."


if __name__ == "__main__":
    args = parse_args()

    # set the background_ref and background_trt
    if args.background_ref[0] == "(":
        args.background_ref = eval(args.background_ref)
    if args.background_trt[0] == "(":
        args.background_trt = eval(args.background_trt)
    # show params
    print(args)

    train(args)
