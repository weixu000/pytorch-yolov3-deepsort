import cv2
import torch

from bbox import center_to_corner, threshold_confidence, NMS
from darknet_parsing import parse_cfg_file, parse_darknet, parse_weights_file
from letterbox import letterbox_image, inv_letterbox_bbox
from util import load_classes, color_map, cvmat_to_tensor, draw_detections


class Detecter:
    """
    Encapsulate YOLO model and the letterbox transform
    """
    classes = load_classes('data/coco.names')
    cmap = color_map(len(classes))
    arch = 'yolov3'

    def __init__(self, inp_dim=None):
        # Set up the neural network
        self.net_info, self.net = parse_darknet(parse_cfg_file(f'cfg/{self.arch}.cfg'))
        parse_weights_file(self.net, f'weights/{self.arch}.weights')
        self.net.cuda().eval()
        print("Network successfully loaded")

        self.inp_dim = inp_dim if inp_dim is not None else self.net_info["inp_dim"][::-1]

    def detect(self, cvmat):
        img = letterbox_image(cvmat, self.inp_dim)
        tensor = cvmat_to_tensor(img).unsqueeze(0).cuda()
        with torch.no_grad():
            output = self.net(tensor).data
        output = tuple(y.cpu() for y in threshold_confidence(output)[0])
        center_to_corner(output[0])
        output = NMS(output)
        inv_letterbox_bbox(output[0], self.inp_dim, cvmat.shape[:2])
        return output


if __name__ == '__main__':
    detecter = Detecter()

    img = cv2.imread('imgs/dog-cycle-car.png')
    output = detecter.detect(img)

    draw_detections(img, output, Detecter.classes, Detecter.cmap)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
