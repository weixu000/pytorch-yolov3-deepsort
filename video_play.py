import cv2
import torch

from detection import Detecter
from sort import Sort
from util import DurationTimer, draw_text, iterate_video, draw_trackers

if __name__ == '__main__':
    cap = cv2.VideoCapture('/home/wei-x15/Downloads/TownCentreXVID.avi')
    orig_dim = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    inp_dim = tuple((x // 4 // 2 ** 5 + 1) * 2 ** 5 for x in orig_dim)
    detecter = Detecter(inp_dim)

    tracker = Sort()

    for frame in iterate_video(cap):
        with DurationTimer() as d:
            output = detecter.detect(frame)
            output = tuple(y[output[1] == Detecter.classes.index('person')] for y in output)  # Select persons
            trks = tracker.update(torch.cat((output[0], output[2].unsqueeze(1)), dim=1).numpy())

        draw_trackers(frame, trks)
        draw_text(frame, f'FPS: {int(1 / d.duration)}', [255, 255, 255], upper_right=(frame.shape[1] - 1, 0))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
