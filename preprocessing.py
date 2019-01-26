import cv2
import torch


def cvmat_to_tensor(mat):
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    mat = mat.transpose((2, 0, 1))
    mat = torch.from_numpy(mat).float().div(255)
    return mat
