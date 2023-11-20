"""
视频读取器: VideoReader
视频写入器: VideoWriter
获取视频时长(单位:秒): get_video_seconds

`视频读取器`基于opencv-python,
支持以帧索引为标识获取视频帧, 当所需索引为当前索引时, 直接read, 否则set+read.
`视频写入器`基于ffmpeg-python,
支持创建"libx264+rgb24的mp4容器"或"libvpx+rgba的webm容器", 支持添加音轨.

"""
import os

import cv2
import ffmpeg
import numpy as np


class VideoReader(object):
    def __init__(self, video_read_path: str) -> None:
        """
        Params:
            video_read_path(str): 视频文件路径;
        """

        self.cap = cv2.VideoCapture(video_read_path)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_conut = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_per_second = round(self.cap.get(cv2.CAP_PROP_FPS), 2)
        self.current_frame_index = 0

    def __len__(self):
        return self.frame_conut

    @property
    def shape(self):
        return (self.frame_height, self.frame_width)

    @property
    def fps(self):
        return self.frame_per_second

    def __getitem__(self, idx):
        if idx == self.current_frame_index:
            still_reading, frame = self.cap.read()
            self.current_frame_index += 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            still_reading, frame = self.cap.read()
            self.current_frame_index = idx + 1

        if not still_reading:
            raise ValueError(f"Video Read Error, at frame index: {idx}")

        return frame

    def close(self):
        self.cap.release()


class VideoWriter(object):
    def __init__(self, video_write_path: str,
                 height: int, width: int, fps: float,
                 pixel_format: str = "rgb24",
                 audio: any = None) -> None:
        """
        Params:
            video_write_path(str): 视频文件保存路径;
            height(int): 输出视频尺寸的高;
            width(int):  输出视频尺寸的宽;
            fps(int): 输出视频帧率;
            pixel_format(str): ["rgb24", "rgba"];
            audio(any): 输出音频;
        """
        self.frame_count = 0
        self.frame_height = height
        self.frame_width = width
        self.frame_per_second = fps
        self.input_pix_fmt = pixel_format

        if pixel_format == "rgba":
            if audio is not None:
                self.writer_process = (
                    ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt=self.input_pix_fmt,
                           s="{}x{}".format(width, height), framerate=fps)
                    .output(audio, video_write_path,
                            vcodec="libvpx", pix_fmt="yuva420p",
                            acodec="libvorbis", loglevel="error",
                            )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
            else:
                self.writer_process = (
                    ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt=self.input_pix_fmt,
                           s="{}x{}".format(width, height), framerate=fps)
                    .output(video_write_path,
                            vcodec="libvpx", pix_fmt="yuva420p",
                            loglevel="error",
                            )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
        elif pixel_format == "rgb24":
            if audio is not None:
                self.writer_process = (
                    ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt=self.input_pix_fmt,
                           s="{}x{}".format(width, height), framerate=fps)
                    .output(audio, video_write_path,
                            vcodec="libx264", pix_fmt="yuv420p",
                            acodec="aac", loglevel="error",
                            )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
            else:
                self.writer_process = (
                    ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt=self.input_pix_fmt,
                           s="{}x{}".format(width, height), framerate=fps)
                    .output(video_write_path,
                            vcodec="libx264", pix_fmt="yuv420p",
                            loglevel="error",
                            )
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
        else:
            raise ValueError(f"{pixel_format} is not support for now.")

    def __len__(self):
        return self.frame_count

    def write(self, image: np.ndarray):
        """
        Params:
            image(np.ndarray): 输入视频帧, 请确保高宽与初始化时的高宽一致,
                色彩空间为`RGB`或`RGBA`;
        """
        assert image.shape[0] == self.frame_height and image.shape[1] == self.frame_width
        if self.input_pix_fmt == "rgb24":
            assert image.shape[2] == 3
        else:
            assert image.shape[2] == 4

        frame = image.astype("uint8").tobytes()
        self.writer_process.stdin.write(frame)
        self.frame_count += 1

    def close(self):
        self.writer_process.stdin.close()
        self.writer_process.wait()


def get_video_seconds(video_path: str):
    """获取视频的时长.

    Params:
        video_path(str): 视频路径.

    Returns:
        float: 视频的时长, 单位为秒.
    """
    assert os.path.exists(video_path), \
        "The input video path does not exist."

    video_reader = VideoReader(video_path)
    video_seconds = float(len(video_reader) / video_reader.fps)
    video_reader.close()

    del video_reader

    return video_seconds


if __name__ == "__main__":
    pass
