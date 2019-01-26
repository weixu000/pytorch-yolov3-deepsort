import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bbox import threshold_confidence, NMS
from coco import COCODataset
from darknet_parsing import parse_cfg_file, parse_darknet, parse_weights_file
from letterbox import letterbox_image
from util import cvmat_to_tensor

if __name__ == '__main__':
    # Setup the neural network
    net_info, net = parse_darknet(parse_cfg_file('cfg/yolov3.cfg'))
    parse_weights_file(net, 'weights/yolov3.weights')
    print("Network successfully loaded")

    inp_dim = net_info["inp_dim"][::-1]


    # Setup coco dataset
    def transform(img_path, bbox, cls):
        orig_img = cv2.imread(img_path)
        img = letterbox_image(orig_img, inp_dim)
        tensor = cvmat_to_tensor(img)

        # bbox = torch.tensor(bbox)
        # cls = torch.tensor(cls)

        # bbox[:, 2] += bbox[:, 0]
        # bbox[:, 3] += bbox[:, 1]

        # return orig_img, img, tensor, bbox, cls
        return tensor


    dataset = COCODataset('COCO/annotations/instances_val2017.json', 'COCO/val2017', transform=transform)
    loader = DataLoader(dataset, batch_size=25, num_workers=4, pin_memory=True)

    out = []
    net.cuda().eval()
    with torch.no_grad():
        for img in tqdm(loader):
            output = net(img.cuda()).data
            output = threshold_confidence(output)
            output = NMS(output)
            out.extend(tuple(y.cpu() for y in x) for x in output)

    out = tuple(out)
    torch.save(out, 'coco-val.pt')
