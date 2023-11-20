"""
获取图像: get_image
调整图像尺寸: image_resize, image_slice

"""
import os
from typing import Union

import cv2
import numpy as np

SUPPORT_COLOR = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "cyan": (255, 255, 0),
    "fuchsia": (255, 0, 255),
    "yellow": (0, 255, 255),
}


def get_image(input: Union[str, tuple, list, np.ndarray],
              height: int = None, width: int = None
              ) -> np.ndarray:
    """获取图像数据.

    Params:
        input(str, tuple, list, numpy.ndarray): 
            str: 文件路径 or 颜色名称;
            tuple or list: 颜色rgb值, (R, G, B);
            numpy.ndarray: 图像数据, 如果输入是array, 则直接输出.
        height(int): 目标图像的高, 仅当`input`不为`array`时生效;
        width(int): 目标图像的宽, 仅当`input`不为`array`时生效;

    Returns:
        output(numpy.ndarray): 图像数据;
            shape(H, W, 3), color(R, G, B), dtype"uint8".
    """
    if isinstance(input, str):
        # (1)`input`是`文件路径`;
        if os.path.isfile(input):
            temp = cv2.imdecode(np.fromfile(input, dtype="uint8"), 1)
            output = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        # (2) `input`是`颜色名称`;
        else:
            assert input in SUPPORT_COLOR, f"Unsupported input color: {input}"
            output = np.ones(shape=(height, width, 3), dtype="uint8"
                             ) * SUPPORT_COLOR[input]
    elif isinstance(input, tuple) or isinstance(input, list):
        # (3) `input`是`颜色值`(R, G, B);
        assert len(input) == 3, f"Unsupported input color: {input}"
        output = np.ones(shape=(height, width, 3), dtype="uint8"
                         ) * np.array(input, dtype="uint8")
    elif isinstance(input, np.ndarray):
        # (4) `input`是`图像数据`;
        output = input.copy()
    else:
        raise ValueError(f"Unsupported input type: {type(input)}")

    return output


def image_resize(input_image: np.ndarray,
                 target_shape: tuple,
                 interpolation: int = cv2.INTER_LINEAR,
                 keep_ratio: bool = False,
                 border_value: tuple = (0, 0, 0)
                 ) -> np.ndarray:
    """调整图像的大小.

    Params:
        input_image(numpy.ndarray): 输入图像;
        target_shape(tuple): 目标图像的大小, (H, W);
        interpolation(int): 插值方法, 默认cv2.INTER_LINEAR, 双线性插值;
        keep_ratio(bool): 是否保持长宽比, 默认False;
            如果为True, 则会将图像缩放到目标图像的最大边, 然后在右侧和下侧填充像素到目标尺寸;
        border_value(tuple): 填充的颜色值, 默认(0, 0, 0)黑色.

    Returns:
        numpy.ndarray: 调整后的图像.
    """
    # 输入图像的高和宽
    input_h, input_w = input_image.shape[:2]
    # 目标图像的高和宽
    target_h, target_w = target_shape
    # 保持长宽比
    if keep_ratio:
        # 计算缩放比例
        scale = min(target_h / input_h, target_w / input_w)
        # 缩放图像
        output_image = cv2.resize(input_image, None, fx=scale, fy=scale,
                                  interpolation=interpolation)
        # 填充图像
        output_h, output_w = output_image.shape[:2]
        pad_top = (target_h - output_h) // 2
        pad_bottom = target_h - output_h - pad_top
        pad_left = (target_w - output_w) // 2
        pad_right = target_w - output_w - pad_left
        output_image = cv2.copyMakeBorder(output_image,
                                          pad_top, pad_bottom,
                                          pad_left, pad_right,
                                          cv2.BORDER_CONSTANT,
                                          value=border_value)
    # 不保持长宽比
    else:
        # 缩放图像
        output_image = cv2.resize(input_image, (target_w, target_h),
                                  interpolation=interpolation)
    return output_image, scale, (pad_top, pad_bottom, pad_left, pad_right)


def image_slice(input_image: np.ndarray,
                pad_num: tuple,
                height: int,
                width: int,
                interpolation=cv2.INTER_NEAREST
                ) -> np.ndarray:
    """调整图像的大小.

    Params:
        input_image(numpy.ndarray): 输入图像;
        pad_num(tuple): 各个方向填充的尺寸(pad_top, pad_bottom, pad_left, pad_right);
        height(int): 目标图像的高;
        width(int): 目标图像的宽;
        interpolation(int): 插值方法, 默认cv2.INTER_LINEAR, 双线性插值;

    Returns:
        numpy.ndarray: 调整后的图像.
    """
    output_image = input_image.copy()

    pad_top = pad_num[0]
    pad_bottom = pad_num[1]
    pad_left = pad_num[2]
    pad_right = pad_num[3]

    h, w = input_image.shape[:2]
    if input_image.ndim == 3:
        output_image = output_image[pad_top:h-pad_bottom,
                                    pad_left:w-pad_right, :]
    elif input_image.ndim == 2:
        output_image = output_image[pad_top:h-pad_bottom,
                                    pad_left:w-pad_right]
    output_image = cv2.resize(output_image, (width, height),
                              interpolation=interpolation)

    return output_image
