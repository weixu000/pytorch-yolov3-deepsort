import numpy as np
import torch
from torch import nn as nn

from models import Darknet, Identity, Route, Shortcut, YOLODetection


def find_option(options, name, default=None, convert=None):
    if default is not None: convert = default.__class__
    ret = options[name] if name in options else default
    try:
        return convert(ret)
    except:
        return ret


activation_factory = {
    'logistic': nn.Sigmoid,
    'relu': lambda: nn.ReLU(inplace=True),
    'leaky': lambda: nn.LeakyReLU(0.1, inplace=True),
    'tanh': nn.Tanh,
    'linear': Identity
}


def parse_net(options):
    return {
        'batch': find_option(options, 'batch', 1),
        'learning_rate': find_option(options, 'learning_rate', .001),
        'momentum': find_option(options, 'momentum', .9),

        'inp_dim': (find_option(options, 'height', 0), find_option(options, 'width', 0)),
        'channel': find_option(options, 'channel', 0)
    }


def parse_conv(options, filters_list):
    filters = find_option(options, 'filters', 1)
    kernel_size = find_option(options, 'size', 1)
    stride = find_option(options, 'stride', 1)
    pad = find_option(options, 'pad', 0)
    padding = find_option(options, 'padding', 0)
    groups = find_option(options, 'groups', 1)
    activation = find_option(options, 'activation', 'logistic')
    batch_normalize = find_option(options, 'batch_normalize', False)
    bias = not batch_normalize
    if pad: padding = kernel_size // 2

    # Add the Convolutional layer
    modules = [nn.Conv2d(filters_list[-1], filters, kernel_size, stride, padding, groups=groups, bias=bias)]

    # Add the BatchNorm Layer
    if batch_normalize:
        modules.append(nn.BatchNorm2d(filters))

    # Add the activation.
    modules.append(activation_factory[activation]())

    filters_list.append(filters)
    return nn.Sequential(*modules)


def parse_route(options, filters_list):
    layers = [int(x.strip()) for x in find_option(options, 'layers', '').split(',')]
    assert layers

    for i in range(len(layers)):
        if layers[i] < 0:
            layers[i] = len(filters_list) + layers[i] - 1

    filters = sum((filters_list[x + 1] for x in layers), 0)
    filters_list.append(filters)

    return Route(layers)


def parse_shortcut(options, filters_list):
    prev = find_option(options, 'from', convert=int)
    activation = find_option(options, 'activation', 'logistic')
    assert activation in activation_factory

    if prev < 0: prev = len(filters_list) + prev - 1

    filters_list.append(filters_list[-1])
    assert filters_list[prev + 1] == filters_list[-1]

    return Shortcut(prev, activation_factory[activation]())


def parse_upsample(options, filters_list):
    stride = find_option(options, 'stride', 2)

    filters_list.append(filters_list[-1])
    return nn.Upsample(scale_factor=stride, mode='bilinear')


def parse_maxpool(options, filters_list):
    stride = find_option(options, 'stride', 1)
    kernel_size = find_option(options, 'size', stride)
    padding = find_option(options, 'padding', kernel_size - 1)

    modules = []

    if padding:
        modules.append(nn.ZeroPad2d((0, padding, 0, padding)))
    modules.append(nn.MaxPool2d(kernel_size, stride))

    filters_list.append(filters_list[-1])
    return nn.Sequential(*modules)


def parse_yolo(options, filters_list):
    mask = [int(x.strip()) for x in find_option(options, 'mask', ).split(',')]

    anchors = [int(x.strip()) for x in find_option(options, 'anchors', '').split(',')]
    anchors = [(anchors[x], anchors[x + 1]) for x in range(0, len(anchors), 2)]
    anchors = [anchors[x] for x in mask]

    classes = find_option(options, 'classes', 20)

    filters_list.append(filters_list[-1])
    return YOLODetection(anchors, classes)


parse_func_factory = {
    'convolutional': parse_conv,
    'upsample': parse_upsample,
    'route': parse_route,
    'shortcut': parse_shortcut,
    'maxpool': parse_maxpool,
    'yolo': parse_yolo,
}


def parse_darknet(options):
    net_info = None

    module_list = nn.ModuleList()

    filters_list = [3]  # Input has 3 channels

    for block in options:
        block_type, block = block
        if block_type == "net":
            assert net_info is None
            net_info = parse_net(block)
        else:
            module_list.append(parse_func_factory[block_type](block, filters_list))

    net = Darknet(module_list)

    return net_info, net


def parse_cfg_file(file_path):
    with open(file_path, 'r') as fp:
        block = None

        for line in fp:
            line = line.strip()
            if not line or line[0] == '#': continue

            if line[0] == "[":  # The start of a new block
                if block:
                    yield block
                block = line[1:-1].strip(), {}
            else:
                key, value = line.split("=")
                block[1][key.strip()] = value.strip()
        if block:
            yield block


def parse_param(fp, param):
    weights = torch.from_numpy(np.fromfile(fp, dtype=np.float32, count=param.numel()))
    weights = weights.view_as(param)
    param.data.copy_(weights)


def parse_weights_file(darknet, file_path):
    with open(file_path, "rb") as fp:
        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)

        # The rest of the values are the weights
        for module in darknet.module_list:
            if isinstance(module, nn.Sequential) and isinstance(module[0], nn.Conv2d):
                if isinstance(module[1], nn.BatchNorm2d):
                    bn = module[1]
                    for x in [bn.bias, bn.weight, bn.running_mean, bn.running_var]:
                        parse_param(fp, x)
                else:
                    parse_param(fp, module[0].bias)

                parse_param(fp, module[0].weight)

        assert len(fp.read()) == 0, f'Weights file {file_path} larger than expected'
