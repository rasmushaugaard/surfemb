from typing import Union

import cv2
import numpy as np
import torch
import albumentations as A

imagenet_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def normalize(img: np.ndarray):  # (h, w, 3) -> (3, h, w)
    mu, std = imagenet_stats
    if img.dtype == np.uint8:
        img = img / 255
    img = (img - mu) / std
    return img.transpose(2, 0, 1).astype(np.float32)


def denormalize(img: Union[np.ndarray, torch.Tensor]):
    mu, std = imagenet_stats
    if isinstance(img, torch.Tensor):
        mu, std = [torch.Tensor(v).type(img.dtype).to(img.device)[:, None, None] for v in (mu, std)]
    return img * std + mu


class Unsharpen(A.ImageOnlyTransform):
    def __init__(self, k_limits=(3, 7), strength_limits=(0., 2.), p=0.5):
        super().__init__()
        self.k_limits = k_limits
        self.strength_limits = strength_limits
        self.p = p

    def apply(self, img, **params):
        if np.random.rand() > self.p:
            return img
        k = np.random.randint(self.k_limits[0] // 2, self.k_limits[1] // 2 + 1) * 2 + 1
        s = k / 3
        blur = cv2.GaussianBlur(img, (k, k), s)
        strength = np.random.uniform(*self.strength_limits)
        unsharpened = cv2.addWeighted(img, 1 + strength, blur, -strength, 0)
        return unsharpened


class DebayerArtefacts(A.ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def apply(self, img, **params):
        if np.random.rand() > self.p:
            return img
        assert img.dtype == np.uint8
        # permute channels before bayering/debayering to cover different bayer formats
        channel_idxs = np.random.permutation(3)
        channel_idxs_inv = np.empty(3, dtype=int)
        channel_idxs_inv[channel_idxs] = 0, 1, 2

        # assemble bayer image
        bayer = np.zeros(img.shape[:2], dtype=img.dtype)
        bayer[::2, ::2] = img[::2, ::2, channel_idxs[2]]
        bayer[1::2, ::2] = img[1::2, ::2, channel_idxs[1]]
        bayer[::2, 1::2] = img[::2, 1::2, channel_idxs[1]]
        bayer[1::2, 1::2] = img[1::2, 1::2, channel_idxs[0]]

        # debayer
        debayer_method = np.random.choice((cv2.COLOR_BAYER_BG2BGR, cv2.COLOR_BAYER_BG2BGR_EA))
        debayered = cv2.cvtColor(bayer, debayer_method)[..., channel_idxs_inv]
        return debayered
