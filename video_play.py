import cv2

from sort import DeepSort
from util import DurationTimer, draw_text, iterate_video, draw_trackers
from yolo import Detecter

if __name__ == '__main__':
    cap = cv2.VideoCapture('/home/wei-x15/Downloads/TownCentreXVID.avi')
    orig_dim = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    inp_dim = tuple((x // 4 // 2 ** 5 + 1) * 2 ** 5 for x in orig_dim)
    detecter = Detecter(inp_dim)

    tracker = DeepSort('weights/ckpt.t7')
    # tracker = Sort()

    for frame in iterate_video(cap):
        with DurationTimer() as d:
            detections = detecter.detect(frame)
            # tracks = tracker.update(detections[0])
            tracks = tracker.update(*detections, frame)

        draw_trackers(frame, tracks)
        draw_text(frame, f'FPS: {int(1 / d.duration)}', [255, 255, 255], upper_right=(frame.shape[1] - 1, 0))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
