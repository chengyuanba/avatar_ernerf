import argparse

from generator import generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input audio path.")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Model config file path or model uid.")
    parser.add_argument("--second_st", type=float, required=False,
                        help="Start second.", default=0)
    parser.add_argument("--second_ed", type=float, required=False,
                        help="End second.", default=-1)
    parser.add_argument("--background_rgb", type=str, required=False,
                        help="Background rgb.", default="(0, 0, 0)")
    parser.add_argument("--with_alpha", action="store_true",
                        help="Output video with alpha or not.")
    parser.set_defaults(with_alpha=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # set the background_ref and background_trt
    if args.background_rgb[0] == "(":
        args.background_rgb = eval(args.background_rgb)
    # show params
    print(args)

    generator(
        audio_path=args.input,
        model_cfg_file=args.config,
        second_start=args.second_st,
        second_end=args.second_ed,
        background_rgb=args.background_rgb,
        with_alpha=args.with_alpha
    )
