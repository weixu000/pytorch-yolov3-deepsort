import cv2
import numpy as np
import torch

from .model import Net


class Extractor:
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net.load_state_dict(torch.load(model_path, map_location=self.device)['net_dict'])
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)

    def __call__(self, img: [np.ndarray]):
        img = np.stack([cv2.resize(x.astype(np.float32), (64, 128)) for x in img])
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img.sub_(self.MEAN[None, :, None, None]).div_(self.STD[None, :, None, None])
        img = img.to(self.device)
        with torch.no_grad():
            feature = self.net(img)
        return feature.cpu().numpy()
