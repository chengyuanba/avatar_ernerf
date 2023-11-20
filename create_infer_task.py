import argparse

from generator import generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-c", "--config", type=str)
    args = parser.parse_args()

    generator(
        audio_path=args.input,
        model_cfg_file=args.config,
        second_start=0,
        second_end=-1,
        background_rgb=(0, 0, 0),
        with_alpha=True
    )
