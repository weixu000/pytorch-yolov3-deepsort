import cv2
import torch

from bbox import threshold_confidence, NMS, draw_bbox
from coco import COCODataset
from darknet_parsing import parse_cfg_file, parse_darknet, parse_weights_file
from preprocessing import cvmat_to_tensor, letterbox_transform, inv_letterbox_transform
from util import color_map

if __name__ == '__main__':
    # Setup the neural network
    net_info, net = parse_darknet(parse_cfg_file('cfg/yolov3.cfg'))
    parse_weights_file(net, 'weights/yolov3.weights')
    print("Network successfully loaded")

    inp_dim = net_info["inp_dim"][::-1]


    # Setup coco dataset
    def transform(img_path, bbox, cls):
        orig_img = cv2.imread(img_path)
        img = letterbox_transform(orig_img, inp_dim)
        tensor = cvmat_to_tensor(img)

        bbox = torch.tensor(bbox)
        cls = torch.tensor(cls)

        bbox[:, 2] += bbox[:, 0]
        bbox[:, 3] += bbox[:, 1]

        return orig_img, img, tensor, torch.tensor(bbox), torch.tensor(cls)


    dataset = COCODataset('COCO/annotations/instances_val2017.json', 'COCO/val2017', transform=transform)

    net.eval()
    with torch.no_grad():
        orig_img, img, tensor, bbox, cls = dataset[0]
        output = net(tensor.unsqueeze(0)).data
    output = threshold_confidence(output)
    output = NMS(output)

    classes = dataset.cls
    cmap = color_map(len(classes))

    draw_bbox(img, output[0], classes, cmap)
    img = inv_letterbox_transform(img, orig_img.shape[:-1])
    cv2.imshow('pred', img)

    draw_bbox(orig_img, (bbox, cls), classes, cmap)
    cv2.imshow('label', orig_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
