import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlbr : array_like
        Bounding box in format `(top left, bottom right)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlbr : ndarray
        Bounding box in format `(top left, bottom right)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlbr, confidence, feature):
        self.tlbr = np.asarray(tlbr, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    @property
    def tlwh(self):
        """Convert bounding box to format `(top left, width, height)`.
        """
        ret = self.tlbr.copy()
        ret[2] -= ret[0]
        ret[3] -= ret[1]

        return ret

    @property
    def xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
