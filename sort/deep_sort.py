import numpy as np

from .deep import Extractor
from .utils import NearestNeighborDistanceMetric, Detection, Tracker


class DeepSort(object):
    def __init__(self, model_path: str):
        self.extractor: Extractor = Extractor(model_path, use_cuda=True)

        max_cosine_distance = 0.2
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker: Tracker = Tracker(metric)

    def update(self, bbox_xyxy: np.ndarray, confidences: np.ndarray, ori_img: np.ndarray):
        # generate detections
        features = self._get_features(bbox_xyxy, ori_img)
        detections = [Detection(b, c, f) for b, c, f in zip(bbox_xyxy, confidences, features)]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue
            outputs.append(np.array([*track.tlbr, track.track_id], dtype=np.int))
        return np.stack(outputs, axis=0) if outputs else np.empty((0, 5))

    def _get_features(self, bbox_xyxy: np.ndarray, ori_img: np.ndarray):
        features = [ori_img[y1:y2, x1:x2] for x1, y1, x2, y2 in bbox_xyxy.astype(np.int)]
        features = self.extractor(features)
        return features
