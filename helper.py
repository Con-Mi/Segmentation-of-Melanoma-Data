import torch
import numpy as np
import cv2

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse



def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou

def jaccard(y_true, y_pred):
    # This does not count the mean
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def dice(y_true, y_pred):
    # this does not count the mean
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def save_model(cust_model, name = "fcn.pt"):
    torch.save(cust_model.state_dict(), name)

def load_model(cust_model, model_dir = "./fcn.pt", map_location_device = "cpu"):
    cust_model.load_state_dict(torch.load(model_dir, map_location = map_location_device))
    cust_model.eval()
    return cust_model

def output_trchtensor_to_numpy(torch_tensor):
    numpy_ndarray = torch_tensor.squeeze(0)
    numpy_ndarray = numpy_ndarray.permute(1, 2, 0)
    numpy_ndarray = numpy_ndarray.squeeze()
    numpy_ndarray = numpy_ndarray.detach().numpy()
    return numpy_ndarray

def mask_overlay(image, mask, color=(0, 1, 0)):
    """
    Helper function to visualize mask on the top of the image
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]    
    return img