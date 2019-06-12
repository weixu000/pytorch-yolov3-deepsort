import time

import cv2
import numpy as np


class DurationTimer:
    def __init__(self):
        self.start, self.end = None, None

    @property
    def duration(self):
        return self.end - self.start

    def __enter__(self):
        self.start, self.end = time.time(), None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        return False


def draw_text(img, label, color, bottom_left=None, upper_right=None):
    t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    if bottom_left is None:
        assert upper_right is not None
        bottom_left = upper_right[0] - t_size[0], upper_right[1] + t_size[1]
    else:
        assert upper_right is None
        upper_right = bottom_left[0] + t_size[0], bottom_left[1] - t_size[1]

    cv2.rectangle(img, bottom_left, upper_right, color, -1)
    cv2.putText(img, label, bottom_left, cv2.FONT_HERSHEY_PLAIN, 1, [255 - c for c in color])


def draw_bbox(img, bbox, label_fn=lambda i: '', color_fn=lambda i: [255, 0, 0]):
    """
    Draw bounding boxes on the image
    """
    for i, b in enumerate(bbox):
        b = tuple(b)
        p1, p2 = b[:2], b[2:]
        cv2.rectangle(img, p1, p2, color_fn(i))

        label = label_fn(i)
        if label:
            draw_text(img, label, color_fn(i), bottom_left=p1)


def draw_detections(img, detections, classes, cmap):
    """
    Draw bounding boxes on the image and add class label and confidence score as title
    """
    bbox, cls, scr = detections
    label_fn = lambda i: f'{classes[cls[i].long().item()]} {scr[i].item():.2f}'
    color_fn = lambda i: cmap[cls[i].long().item()]
    draw_bbox(img, bbox.long().numpy(), label_fn, color_fn)


def draw_trackers(img, trackers):
    if trackers.shape[0] == 0: return
    bbox, id = trackers[:, :-1], trackers[:, -1]
    label_fn = lambda i: f'{int(id[i])}'
    draw_bbox(img, bbox.astype(np.int), label_fn, lambda i: [0, 0, 255])


def iterate_video(capture):
    while capture.isOpened():
        retval, frame = capture.read()
        if retval:
            yield frame
        else:
            break
