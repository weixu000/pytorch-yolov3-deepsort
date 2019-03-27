"""
Pytorch models of YOLO
"""

import torch
from torch import nn as nn

from .bbox import anchor_transform


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Route(nn.Module):
    def __init__(self, layers):
        super(Route, self).__init__()
        self.layers = layers

    def forward(self, *inputs):
        assert len(self.layers) == len(inputs)
        return torch.cat(inputs, dim=1)


class Shortcut(nn.Module):
    def __init__(self, prev, activation):
        super(Shortcut, self).__init__()
        self.prev = prev
        self.activation = activation

    def forward(self, *inputs):
        assert len(inputs) == 2
        return self.activation(inputs[0] + inputs[1])


class YOLODetection(nn.Module):
    def __init__(self, anchors):
        super(YOLODetection, self).__init__()
        self.register_buffer('anchors', torch.tensor(anchors, dtype=torch.float))
        self.grid = [torch.empty(0), torch.empty(0)]

    def forward(self, x, inp_dim):
        grid_size = x.shape[2:]
        if grid_size != [x.shape[0] for x in self.grid]:
            self.grid = [torch.arange(grid_size[0], dtype=x.dtype, device=x.device),
                         torch.arange(grid_size[1], dtype=x.dtype, device=x.device)]

        batch_size = x.shape[0]
        stride = inp_dim[0] // grid_size[0], inp_dim[1] // grid_size[1]
        num_anchors = len(self.anchors)
        bbox_attrs = x.shape[1] // num_anchors

        x = x.view(batch_size, num_anchors, bbox_attrs, *grid_size)

        return anchor_transform(x, self.anchors, self.grid, stride)


class Darknet(nn.Module):
    def __init__(self, module_list):
        super(Darknet, self).__init__()

        self.module_list = module_list
        self.need_cache = set()
        self.prepare_cache()

    def forward(self, x):
        input_dim = x.shape[2:4]
        detections = []
        cache = {}  # We cache the outputs for the route layer

        for i, module in enumerate(self.module_list):
            if isinstance(module, Route):
                x = module(*[cache[j] for j in module.layers])
            elif isinstance(module, Shortcut):
                x = module(x, cache[module.prev])
            elif isinstance(module, YOLODetection):
                x = module(x, input_dim)
                detections.append(x)
            else:
                x = module(x)
            if i in self.need_cache:
                cache[i] = x

        return torch.cat(detections, dim=1)

    def prepare_cache(self):
        self.need_cache.clear()
        for i, module in enumerate(self.module_list):
            if isinstance(module, Route):
                self.need_cache.update(module.layers)
            elif isinstance(module, Shortcut):
                self.need_cache.add(module.prev)
