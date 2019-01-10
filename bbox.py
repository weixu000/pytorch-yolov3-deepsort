import cv2
import torch


def IOU(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = [x.unsqueeze(0) for x in (box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3])]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    assert torch.all(b1_x2 > b1_x1) and torch.all(b1_y2 > b1_y1) and torch.all(b2_x2 > b2_x1) and torch.all(
        b2_y2 > b2_y1)

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


def draw_bbox(img, bbox_batch, classes=None, cmap=None):
    """
    Draw bounding boxes on the image and add class label and confidence score as title
    """
    for bbox in zip(*bbox_batch):
        if len(bbox) == 2:  # Without scores
            bbox, cls = bbox
            bbox, cls = tuple(bbox.long().tolist()), cls.long().item()
            label = f'{classes[cls] if classes is not None else cls}'
        else:
            bbox, cls, scr = bbox
            bbox, cls, scr = tuple(bbox.long().tolist()), cls.long().item(), scr.item()
            label = f'{classes[cls] if classes is not None else cls} {scr:.2f}'

        color = cmap[cls] if cmap is not None else [255, 0, 0]
        p1, p2 = bbox[:2], bbox[2:]
        cv2.rectangle(img, p1, p2, color)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        p2 = p1[0] + t_size[0], p1[1] - t_size[1]
        cv2.rectangle(img, p1, p2, color, -1)
        cv2.putText(img, label, p1, cv2.FONT_HERSHEY_PLAIN, 1, [255 - c for c in color])


def anchor_transform(prediction, inp_dim, anchors, num_classes):
    batch_size = prediction.shape[0]
    grid_size = prediction.shape[2:]
    stride = inp_dim[0] // grid_size[0], inp_dim[1] // grid_size[1]
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, num_anchors, bbox_attrs, *grid_size)

    # Sigmoid object confidence
    prediction[:, :, 4].sigmoid_()

    # Softmax the class scores
    prediction[:, :, 5:] = prediction[:, :, 5:].softmax(-1)

    # Sigmoid the centre_X, centre_Y
    grid = torch.arange(grid_size[1]).to(prediction)
    prediction[:, :, 0].sigmoid_().add_(grid.view(1, 1, 1, -1)).mul_(stride[1])
    grid = torch.arange(grid_size[0]).to(prediction)
    prediction[:, :, 1].sigmoid_().add_(grid.view(1, 1, -1, 1)).mul_(stride[0])

    # log space transform height and the width
    anchors = prediction.new_tensor(anchors)
    prediction[:, :, 2].exp_().mul_(anchors[:, 0].view(1, -1, 1, 1))
    prediction[:, :, 3].exp_().mul_(anchors[:, 1].view(1, -1, 1, 1))

    return prediction.transpose(2, -1).contiguous().view(batch_size, -1, bbox_attrs)


def center_to_corner(pred):
    """
    Convert (center_x, center_y, width, height) to (topleft_x, topleft_y, bottomleft_x, bottomright_y)
    """
    pred[:, :, 0] -= pred[:, :, 2] / 2
    pred[:, :, 1] -= pred[:, :, 3] / 2
    pred[:, :, 2] += pred[:, :, 0]
    pred[:, :, 3] += pred[:, :, 1]


def threshold_confidence(pred, threshold=0.1):
    """
    Threshold bounding boxes by probability
    Returns a list of (corners of boxes, class label, probability) for each image
    """
    center_to_corner(pred)
    max_conf_score, max_conf = pred[:, :, 5:].max(2)
    max_conf_score *= pred[:, :, 4]  # probability = object confidence * class score
    prob_thresh = max_conf_score > threshold

    output = []
    for batch in zip(pred[:, :, :4], max_conf, max_conf_score, prob_thresh):
        output.append(tuple(x[batch[-1]] for x in batch[:-1]))

    return tuple(output)


def NMS(preds, threshold=0.4):
    """
    Non maximal suppression
    """
    output = []
    for bbox_batch, cls_batch, scr_batch in preds:
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
            ind_batch.append(ind_cls)
        ind_batch = torch.cat(ind_batch)
        output.append(tuple(x[ind_batch] for x in [bbox_batch, cls_batch, scr_batch]))
    return tuple(output)
