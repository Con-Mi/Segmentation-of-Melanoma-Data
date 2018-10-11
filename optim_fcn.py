import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from fcn_naive_model import fcn_model
from data_loader import Melanoma_Train_Validation_DataLoader

import time
import copy

use_cuda = torch.cuda.is_available()

# Hyperparameters
batch_size = 16
nr_epochs = 10
momentum = 0.9
learning_rate = 0.01
running_loss = 0.0
gamma = 0.1
milestones = [1, 2, 3, 5, 7, 8]
img_size = 512

segm_model = fcn_model(is_pretrained = True)
if use_cuda:
    segm_model.cuda()

transform = [ transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p = 1.0), 
             transforms.RandomRotation(205), transforms.RandomRotation(45), transforms.RandomRotation(145), 
             transforms.RandomRotation(300), transforms.ColorJitter(brightness=1.3), transforms.ColorJitter(contrast=1.2),
             transforms.ColorJitter(saturation=1.2), transforms.ColorJitter(saturation=0.7), transforms.ColorJitter(hue=0.3),
             transforms.ColorJitter(hue=0.1) ]
# Note: FIX AUGMENTATIONS
train_loader, validation_loader = Melanoma_Train_Validation_DataLoader(batch_size = batch_size, data_transforms=transforms.Compose([transforms.Resize([img_size, img_size]), transforms.ToTensor()]))


optimizer = optim.SGD(segm_model.parameters(), lr = learning_rate, momentum = momentum)
criterion = nn.BCEWithLogitsLoss().cuda() if use_cuda else nn.BCEWithLogitsLoss()

dataloader_dict = {"train": train_loader, "valid": validation_loader}

def train_model(cust_model, dataloaders, criterion, optimizer, num_epochs = 10, scheduler = None):
    start_time = time.time()
    val_acc_history = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("_"*15)
        for phase in ["train", "valid"]:
            if phase == "train":
                cust_model.train()
            if phase == "valid":
                cust_model.eval()
            running_loss = 0.0
            # running_corrects = 0

            for input_img, labels in dataloaders[phase]:
                input_img = input_img.cuda() if use_cuda else input_img
                labels = labels.cuda() if use_cuda else labels

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    outputs = cust_model(input_img)
                    loss = criterion(outputs, labels)
                    # _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input_img.size(0)
        
            epoch_loss = running_loss / len(dataloaders[phase])
            # epoch_acc calculate accuracy from Jaccard loss
            epoch_acc = 0.0

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            if phase == "valid" and epoch_acc > best_acc:
                # best_acc = epoc_acc
                # best_model_wts = copy.deepcopy(cust_model.state_dict)
                pass
            if phase == "valid":
                # val_acc_history.append(epoch_acc)
                pass
        print()
    time_elapsed = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best validation Accuracy: {:.4f}".format(best_acc))
    best_model_wts = copy.deepcopy(cust_model.state_dict())   # Need to change this in the future when I fix the jaccard index
    cust_model.load_state_dict(best_model_wts)
    return cust_model, val_acc_history

def save_model(cust_model, name = "fcn.pt"):
    torch.save(cust_model.state_dict(), name)

def load_model(cust_model, model_dir = "./fcn.pt"):
    cust_model.load_state_dict(torch.load(model_dir))
    cust_model.eval()
    return cust_model

segm_model, acc = train_model(segm_model, dataloader_dict, criterion, optimizer, nr_epochs)


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



