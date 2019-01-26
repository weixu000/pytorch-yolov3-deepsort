import cv2
import torch

from bbox import draw_bbox, inv_letterbox_bbox
from darknet_parsing import parse_cfg_file, parse_darknet, parse_weights_file
from detection import detect
from preprocessing import cvmat_to_tensor, letterbox_image
from util import load_classes, color_map, DurationTimer, draw_text

if __name__ == '__main__':
    # Set up the neural network
    net_info, net = parse_darknet(parse_cfg_file('cfg/yolov3.cfg'))
    parse_weights_file(net, 'weights/yolov3.weights')
    print("Network successfully loaded")

    # inp_dim = net_info["inp_dim"][::-1]

    cap = cv2.VideoCapture(r'D:\Downloads\TownCentreXVID.avi')
    orig_dim = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    inp_dim = tuple((x // 4 // 2 ** 5 + 1) * 2 ** 5 for x in orig_dim)

    classes = load_classes('data/coco.names')
    cmap = color_map(len(classes))

    net.cuda().eval()
    with torch.no_grad():
        while cap.isOpened():
            _, frame = cap.read()

            with DurationTimer() as d:
                output = detect(net, cvmat_to_tensor(letterbox_image(frame, inp_dim)).cuda())
                output = tuple(y[output[1] == classes.index('person')] for y in output)  # Select persons
                inv_letterbox_bbox(output[0], inp_dim, orig_dim)
                output = tuple(y.cpu() for y in output)

            draw_bbox(frame, output, classes, cmap)
            draw_text(frame, f'FPS: {int(1 / d.duration)}', [255, 255, 255], upper_right=(frame.shape[1] - 1, 0))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
