import cv2
import numpy as np
import torch


def letterbox_dim(orig, box):
    h, w = box
    img_h, img_w = orig

    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))

    return new_h, new_w


def letterbox_transform(img, box_dim):
    """resize image with unchanged aspect ratio using padding"""
    h, w = box_dim
    new_h, new_w = letterbox_dim(img.shape[:-1], box_dim)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    out = np.full((h, w, 3), 128, dtype=np.uint8)
    out[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = img

    return out


def inv_letterbox_transform(box, img_dim):
    img_h, img_w = img_dim
    h, w = box.shape[:-1]

    new_h, new_w = letterbox_dim(img_dim, box.shape[:-1])

    box = box[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :]
    box = cv2.resize(box, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    return box


def cvmat_to_tensor(mat):
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    mat = mat.transpose((2, 0, 1))
    mat = torch.from_numpy(mat).float().div(255)
    return mat
