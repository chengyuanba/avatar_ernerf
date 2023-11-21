import os

import numpy as np
import torch
from tqdm import tqdm


def load_dir(path, start, end):
    lmks = []
    imgs_paths = []
    for i in tqdm(range(start, end), desc="loading data"):
        img_path = os.path.join(path, f"{i}.jpg")
        lmk_path = os.path.join(path, f"{i}.lms")
        if os.path.exists(img_path) and os.path.exists(lmk_path):
            lmk = np.loadtxt(lmk_path, dtype="float32")
            lmks.append(lmk)
            imgs_paths.append(img_path)
    lmks = np.stack(lmks)
    lmks = torch.as_tensor(lmks).cuda()
    
    return lmks, imgs_paths