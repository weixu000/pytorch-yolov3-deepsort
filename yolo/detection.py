import cv2
import torch

from .bbox import center_to_corner, threshold_confidence, NMS
from .darknet_parsing import parse_cfg_file, parse_darknet, parse_weights_file
from .letterbox import letterbox_image, inv_letterbox_bbox


def load_classes(file_path):
    with open(file_path, "r") as fp:
        names = [x for x in fp.read().split("\n") if x]
    return names


def color_map(n):
    def bit_get(x, i):
        return x & (1 << i)

    cmap = []
    for i in range(n):
        r = g = b = 0
        for j in range(7, -1, -1):
            r |= bit_get(i, 0) << j
            g |= bit_get(i, 1) << j
            b |= bit_get(i, 2) << j
            i >>= 3

        cmap.append((r, g, b))
    return cmap


def cvmat_to_tensor(mat):
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    mat = mat.transpose((2, 0, 1))
    mat = torch.from_numpy(mat).float().div(255)
    return mat


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

        box, cls, scr = output
        ind = cls == self.classes.index('person')
        box, scr = box[ind], scr[ind]

        box[:, [0, 2]] = torch.clamp(box[:, [0, 2]], min=0, max=cvmat.shape[1])
        box[:, [1, 3]] = torch.clamp(box[:, [1, 3]], min=0, max=cvmat.shape[0])

        return box.numpy(), scr.numpy()
