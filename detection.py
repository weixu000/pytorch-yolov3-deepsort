import cv2
import torch

from bbox import threshold_confidence, NMS, draw_bbox
from darknet_parsing import parse_cfg_file, parse_darknet, parse_weights_file
from preprocessing import cvmat_to_tensor, letterbox_transform, inv_letterbox_transform
from util import load_classes, color_map


def detect(net, tensor):
    net.eval()
    with torch.no_grad():
        output = net(tensor.unsqueeze(0)).data
        output = threshold_confidence(output)
        return NMS(output)[0]


if __name__ == '__main__':
    # Set up the neural network
    net_info, net = parse_darknet(parse_cfg_file('cfg/yolov3.cfg'))
    parse_weights_file(net, 'weights/yolov3.weights')
    print("Network successfully loaded")

    inp_dim = net_info["inp_dim"][::-1]

    orig_img = cv2.imread('imgs/dog-cycle-car.jpg')
    img = letterbox_transform(orig_img, inp_dim)

    output = detect(net, cvmat_to_tensor(img))

    classes = load_classes('data/coco.names')
    cmap = color_map(len(classes))
    draw_bbox(img, output, classes, cmap)

    img = inv_letterbox_transform(img, orig_img.shape[:-1])

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
