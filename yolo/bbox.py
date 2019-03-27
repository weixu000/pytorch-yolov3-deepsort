"""
Algorithms on bounding boxes
"""
import torch


def IOU(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = [x.unsqueeze(1) for x in box1.transpose(0, 1)]
    b2_x1, b2_y1, b2_x2, b2_y2 = [x.unsqueeze(0) for x in box2.transpose(0, 1)]

    # get the corrdinates of the intersection rectangle
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def anchor_transform(prediction, anchors, grid, stride):
    """
    Transform network prediction into (center_x, center_y, width, height)
    """
    # Sigmoid object confidence
    prediction[:, :, 4].sigmoid_()

    # Softmax the class scores
    prediction[:, :, 5:] = prediction[:, :, 5:].softmax(-1)

    # Sigmoid the centre_X, centre_Y
    prediction[:, :, 0].sigmoid_().add_(grid[1].view(1, 1, 1, -1)).mul_(stride[1])
    prediction[:, :, 1].sigmoid_().add_(grid[0].view(1, 1, -1, 1)).mul_(stride[0])

    # log space transform height and the width
    prediction[:, :, 2].exp_().mul_(anchors[:, 0].view(1, -1, 1, 1))
    prediction[:, :, 3].exp_().mul_(anchors[:, 1].view(1, -1, 1, 1))

    return prediction.transpose(2, -1).contiguous().view(prediction.shape[0], -1, prediction.shape[2])


def center_to_corner(bbox):
    """
    Convert (center_x, center_y, width, height) to (topleft_x, topleft_y, bottomleft_x, bottomright_y)
    """
    bbox[:, 0] -= bbox[:, 2] / 2
    bbox[:, 1] -= bbox[:, 3] / 2
    bbox[:, 2] += bbox[:, 0]
    bbox[:, 3] += bbox[:, 1]


def threshold_confidence(pred, threshold=0.1):
    """
    Threshold bounding boxes by probability
    :return ((corners of boxes, class label, probability) for each image)
    """
    max_cls_score, max_cls = pred[:, :, 5:].max(2)
    max_cls_score *= pred[:, :, 4]  # probability = object confidence * class score
    prob_thresh = max_cls_score > threshold

    output = []
    for batch in zip(pred[:, :, :4], max_cls, max_cls_score, prob_thresh):
        output.append(tuple(x[batch[-1]] for x in batch[:-1]))

    return tuple(output)


def NMS(batch, threshold=0.4):
    """
    Non maximal suppression
    :return (corners of boxes, class label, probability)
    """
    bbox_batch, cls_batch, scr_batch = batch
    ind_batch = []
    for cls in cls_batch.unique():
        ind_cls = (cls_batch == cls).nonzero().squeeze(1)
        bbox_cls, scr_cls = bbox_batch[ind_cls], scr_batch[ind_cls]

        scr_cls, sorted_ind = scr_cls.sort(descending=True)
        ind_cls = ind_cls[sorted_ind]
        bbox_cls = bbox_cls[sorted_ind]

        i = 0
        while i < ind_cls.shape[0]:
            # Get IOUs of all boxes coming after bbox_cls[i]
            ious = IOU(bbox_cls[i].unsqueeze(0), bbox_cls[i + 1:]).squeeze(0)

            # Move boxes with smaller IOUs up and remove the others
            iou_ind = (ious < threshold).nonzero().squeeze(1)

            ind_cls[i + 1:i + 1 + iou_ind.shape[0]] = ind_cls[i + 1 + iou_ind]
            bbox_cls[i + 1:i + 1 + iou_ind.shape[0]] = bbox_cls[i + 1 + iou_ind]

            ind_cls = ind_cls[:i + 1 + iou_ind.shape[0]]
            bbox_cls = bbox_cls[:i + 1 + iou_ind.shape[0]]
            i += 1
        ind_batch += ind_cls.tolist()
    return tuple(x[ind_batch] for x in [bbox_batch, cls_batch, scr_batch])
