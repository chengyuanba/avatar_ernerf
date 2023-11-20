import argparse
import time
import warnings
from queue import Queue
from threading import Event, Thread

import numpy as np
import pyaudio
import librosa
import torch
import torch.nn.functional as F
from transformers import (Data2VecAudioModel, HubertModel,
                          Wav2Vec2FeatureExtractor, Wav2Vec2Model)

warnings.filterwarnings("ignore")


def load_model_Data2VecAudio(model_name_or_path, device):
    print(f"===>>> Loading Data2VecAudio model from: {model_name_or_path}")
    print(f"===>>> Using device: {device}.")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path)
    model = Data2VecAudioModel.from_pretrained(model_name_or_path)
    model.to(device)
    return processor, model


def load_model_Hubert(model_name_or_path, device):
    print(f"===>>> Loading Hubert model from: {model_name_or_path}")
    print(f"===>>> Using device: {device}.")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path)
    model = HubertModel.from_pretrained(model_name_or_path)
    model.to(device)
    return processor, model


def load_model_Wav2Vec2(model_name_or_path, device):
    print(f"===>>> Loading Wav2Vec2 model from: {model_name_or_path}")
    print(f"===>>> Using device: {device}.")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path)
    model = Wav2Vec2Model.from_pretrained(model_name_or_path)
    model.to(device)
    return processor, model


def _read_frame(stream, exit_event, queue, chunk):

    while True:
        if exit_event.is_set():
            break
        frame = stream.read(chunk, exception_on_overflow=False)
        frame = np.frombuffer(frame, dtype=np.int16).astype(
            np.float32) / 32767  # [chunk]
        queue.put(frame)


def _play_frame(stream, exit_event, queue, chunk):

    while True:
        if exit_event.is_set():
            break
        frame = queue.get()
        frame = (frame * 32767).astype(np.int16).tobytes()
        stream.write(frame, chunk)


class ASR:
    def __init__(self, opt):

        # Init params
        self.opt = opt
        self.play = opt.asr_play

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fps = opt.fps  # 20 ms per frame
        self.sample_rate = 16000
        # 320 samples per chunk (20ms * 16000 / 1000)
        self.chunk = self.sample_rate // self.fps
        self.mode = "live" if opt.asr_wav == "" else "file"

        # Load model
        if opt.model_name.lower() == "data2vecaudio":
            self.processor, self.model = load_model_Data2VecAudio(
                opt.model_weight, self.device)
        elif opt.model_name == "hubert":
            self.processor, self.model = load_model_Hubert(
                opt.model_weight, self.device)
        elif opt.model_name == "wav2vec2":
            self.processor, self.model = load_model_Wav2Vec2(
                opt.model_weight, self.device)
        else:
            raise ValueError(f"Unknown model_name: {opt.model_name}.")

        # Get feature ndim
        if opt.model_arch == "base":
            self.audio_dim = 768
        elif opt.model_arch == "large":
            self.audio_dim = 1024
        else:
            raise ValueError(f"Unknown model_arch: {opt.model_arch}.")

        # prepare context cache
        # each segment is (stride_left + ctx + stride_right) * 20ms, latency should be (ctx + stride_right) * 20ms
        self.context_size = opt.m
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        self.terminated = False
        self.frames = []

        # pad left frames
        if self.stride_left_size > 0:
            self.frames.extend(
                [np.zeros(self.chunk, dtype=np.float32)] * self.stride_left_size)

        self.exit_event = Event()
        self.audio_instance = pyaudio.PyAudio()

        # create input stream
        if self.mode == "file":
            self.file_stream = self.create_file_stream()
        else:
            # start a background process to read frames
            self.input_stream = self.audio_instance.open(
                format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, output=False, frames_per_buffer=self.chunk)
            self.queue = Queue()
            self.process_read_frame = Thread(target=_read_frame, args=(
                self.input_stream, self.exit_event, self.queue, self.chunk))

        # play out the audio too...?
        if self.play:
            self.output_stream = self.audio_instance.open(
                format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
                input=False, output=True, frames_per_buffer=self.chunk)
            self.output_queue = Queue()
            self.process_play_frame = Thread(target=_play_frame, args=(
                self.output_stream, self.exit_event, self.output_queue, self.chunk))

        # current location of audio
        self.idx = 0

        # prepare to save logits
        if self.opt.asr_save_feats:
            self.all_feats = []

        # the extracted features
        # use a loop queue to efficiently record endless features: [f--t---][-------][-------]
        self.feat_buffer_size = 4
        self.feat_buffer_idx = 0
        self.feat_queue = torch.zeros(
            self.feat_buffer_size * self.context_size, self.audio_dim, dtype=torch.float32, device=self.device)

        # TODO: hard coded 16 and 8 window size...
        self.front = self.feat_buffer_size * self.context_size - 8  # fake padding
        self.tail = 8
        # attention window...
        # 4 zero padding...
        self.att_feats = [torch.zeros(
            self.audio_dim, 16, dtype=torch.float32, device=self.device)] * 4

        # warm up steps needed: mid + right + window_size + attention_size
        self.warm_up_steps = self.context_size + self.stride_right_size + 8 + 2 * 3

        self.listening = False
        self.playing = False

    def listen(self):
        # start
        if self.mode == "live" and not self.listening:
            self.process_read_frame.start()
            self.listening = True

        if self.play and not self.playing:
            self.process_play_frame.start()
            self.playing = True

    def stop(self):

        self.exit_event.set()

        if self.play:
            self.output_stream.stop_stream()
            self.output_stream.close()
            if self.playing:
                self.process_play_frame.join()
                self.playing = False

        if self.mode == "live":
            self.input_stream.stop_stream()
            self.input_stream.close()
            if self.listening:
                self.process_read_frame.join()
                self.listening = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def get_next_feat(self):
        # return a [1/8, 16] window, for the next input to nerf side.

        while len(self.att_feats) < 8:
            # [------f+++t-----]
            if self.front < self.tail:
                feat = self.feat_queue[self.front:self.tail]
            # [++t-----------f+]
            else:
                feat = torch.cat(
                    [self.feat_queue[self.front:], self.feat_queue[:self.tail]], dim=0)

            self.front = (self.front + 2) % self.feat_queue.shape[0]
            self.tail = (self.tail + 2) % self.feat_queue.shape[0]

            self.att_feats.append(feat.permute(1, 0))

        att_feat = torch.stack(self.att_feats, dim=0)  # [8, 44, 16]

        # discard old
        self.att_feats = self.att_feats[1:]

        return att_feat

    def run_step(self):

        if self.terminated:
            return

        # get a frame of audio
        frame = self.get_audio_frame()

        # the last frame
        if frame is None:
            # terminate, but always run the network for the left frames
            self.terminated = True
        else:
            self.frames.append(frame)
            # put to output
            if self.play:
                self.output_queue.put(frame)
            # context not enough, do not run network.
            if len(self.frames) < self.stride_left_size + self.context_size + self.stride_right_size:
                return

        inputs = np.concatenate(self.frames)  # [N * chunk]

        # discard the old part to save memory
        if not self.terminated:
            self.frames = self.frames[-(self.stride_left_size +
                                        self.stride_right_size):]

        feats = self.frame_to_text(inputs)
        # save feats
        if self.opt.asr_save_feats:
            self.all_feats.append(feats)

        # record the feats efficiently.. (no concat, constant memory)
        if not self.terminated:
            start = self.feat_buffer_idx * self.context_size
            end = start + feats.shape[0]
            self.feat_queue[start:end] = feats
            self.feat_buffer_idx = (
                self.feat_buffer_idx + 1) % self.feat_buffer_size

        # will only run once at ternimation
        if self.terminated:
            if self.opt.asr_save_feats:
                feats = torch.cat(self.all_feats, dim=0)  # [N, C]
                window_size = 16
                padding = window_size // 2

                feats = feats.view(-1, self.audio_dim).permute(
                    1, 0).contiguous()  # [C, M]

                feats = feats.view(1, self.audio_dim, -1, 1)  # [1, C, M, 1]

                unfold_feats = F.unfold(
                    feats, kernel_size=(window_size, 1),
                    padding=(padding, 0), stride=(2, 1))  # [1, C * window_size, M / 2 + 1]

                unfold_feats = unfold_feats.view(self.audio_dim, window_size, -1).permute(
                    2, 1, 0).contiguous()  # [C, window_size, M / 2 + 1] --> [M / 2 + 1, window_size, C]

                np.save(self.opt.asr_save_feats_path,
                        unfold_feats.cpu().numpy())

    def create_file_stream(self):

        stream, sample_rate = librosa.load(
            self.opt.asr_wav, sr=self.sample_rate, mono=True)
        stream = stream.astype("float32")

        return stream

    def create_pyaudio_stream(self):

        audio = pyaudio.PyAudio()

        # get devices
        info = audio.get_host_api_info_by_index(0)
        n_devices = info.get("deviceCount")

        for i in range(0, n_devices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels")) > 0:
                name = audio.get_device_info_by_host_api_device_index(0, i).get("name")
                break

        # get stream
        stream = audio.open(input_device_index=i,
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=self.sample_rate,
                            input=True,
                            frames_per_buffer=self.chunk)

        return audio, stream

    def get_audio_frame(self):

        if self.mode == "file":

            if self.idx < self.file_stream.shape[0]:
                frame = self.file_stream[self.idx: self.idx + self.chunk]
                self.idx = self.idx + self.chunk
                return frame
            else:
                return None

        else:

            frame = self.queue.get()
            self.idx = self.idx + self.chunk

            return frame

    def frame_to_text(self, frame):
        # frame: [N * 320], N = (context_size + 2 * stride_size)

        # preprocess
        inputs = self.processor(frame, sampling_rate=self.sample_rate,
                                padding=True, return_tensors="pt")
        # extract features
        with torch.no_grad():
            outputs = self.model(inputs.input_values.to(self.device))
        logits = outputs.last_hidden_state

        # cut off stride
        left = max(0, self.stride_left_size)
        # +1 to make sure output is the same length as input.
        right = min(logits.shape[1],
                    logits.shape[1] - self.stride_right_size + 1)
        # do not cut right if terminated.
        if self.terminated:
            right = logits.shape[1]
        logits = logits[:, left:right]

        return logits[0]

    def run(self):

        self.listen()

        while not self.terminated:
            self.run_step()

    def clear_queue(self):
        # clear the queue, to reduce potential latency...
        if self.mode == "live":
            self.queue.queue.clear()
        if self.play:
            self.output_queue.queue.clear()

    def warm_up(self):
        self.listen()

        t = time.time()
        for _ in range(self.warm_up_steps):
            self.run_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time() - t

        self.clear_queue()


def audio_encoding(input_path: str, output_path: str,
                   model_name: str = "hubert",
                   model_arch: str = "large",
                   model_weight: str = "facebook/hubert-large-ls960-ft"):
    """
    Params:
        input_path(str): 输入音频文件路径;
        output_path(str): 输出特征文件路径;
        model_name(str): 模型名称, ["Data2VecAudio", "Hubert", "Wav2Vec2"];
        model_arch(str): 模型结构, ["base", "large"];
        model_weight(str): 模型权重, HuggingFace ModelName or dir path;

    """
    opt = argparse.Namespace()
    opt.asr_wav: str = input_path
    opt.asr_play: bool = False

    opt.model_name: str = model_name
    opt.model_arch: str = model_arch
    opt.model_weight: str = model_weight

    opt.asr_save_feats: bool = True
    opt.asr_save_feats_path: str = output_path

    opt.fps: int = 50
    opt.m: int = 50
    opt.l: int = 10
    opt.r: int = 10

    with ASR(opt) as asr:
        asr.run()

    data = np.load(output_path)

    return tuple(data.shape)


if __name__ == "__main__":
    pass
