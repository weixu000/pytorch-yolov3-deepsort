import cv2

import numpy as np


def letterbox_dim(orig, box):
    h, w = box
    img_h, img_w = orig

    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))

    return new_h, new_w


def letterbox_image(img, box_dim):
    """resize image with unchanged aspect ratio using padding"""
    h, w = box_dim
    new_h, new_w = letterbox_dim(img.shape[:-1], box_dim)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    out = np.full((h, w, 3), 128, dtype=np.uint8)
    out[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = img

    return out


def inv_letterbox_image(box, img_dim):
    img_h, img_w = img_dim
    h, w = box.shape[:-1]

    new_h, new_w = letterbox_dim(img_dim, box.shape[:-1])

    box = box[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :]
    box = cv2.resize(box, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    return box


def inv_letterbox_bbox(bbox, box_dim, img_dim):
    img_h, img_w = img_dim
    h, w = box_dim
    new_h, new_w = letterbox_dim(img_dim, box_dim)

    bbox[:, [0, 2]] -= (w - new_w) // 2
    bbox[:, [0, 2]] *= img_w / w

    bbox[:, [1, 3]] -= (h - new_h) // 2
    bbox[:, [1, 3]] *= img_h / h