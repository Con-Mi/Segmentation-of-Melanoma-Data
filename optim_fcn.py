import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from fcn_naive_model import fcn_model
from data_loader import Melanoma_Train_Validation_DataLoader
from helper import jaccard, dice, save_model

import time
import copy

use_cuda = torch.cuda.is_available()

# Hyperparameters
batch_size = 10
nr_epochs = 15
momentum = 0.91
learning_rate = 0.01
gamma = 0.1
milestones = [1, 2, 3, 5, 7, 8]
img_size = 512

segm_model = fcn_model(is_pretrained = True)
if use_cuda:
    segm_model.cuda()

mul_transform = [ transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p = 1.0), 
             transforms.RandomRotation(205), transforms.RandomRotation(45), transforms.RandomRotation(145), 
             transforms.RandomRotation(300), transforms.ColorJitter(brightness=1.3), transforms.ColorJitter(contrast=1.2),
             transforms.ColorJitter(saturation=1.2), transforms.ColorJitter(saturation=0.7), transforms.ColorJitter(hue=0.3),
             transforms.ColorJitter(hue=0.1) ]
# Note: FIX AUGMENTATIONS
train_loader, validation_loader = Melanoma_Train_Validation_DataLoader(batch_size = batch_size, data_transforms=transforms.Compose([mul_transform[0], mul_transform[2], mul_transform[6], transforms.Resize([img_size, img_size]), transforms.ToTensor()]))


optimizer = optim.SGD(segm_model.parameters(), lr = learning_rate, momentum = momentum)
criterion = nn.BCEWithLogitsLoss().cuda() if use_cuda else nn.BCEWithLogitsLoss()

dataloader_dict = {"train": train_loader, "valid": validation_loader}

def train_model(cust_model, dataloaders, criterion, optimizer, num_epochs = 10, scheduler = None):
    start_time = time.time()
    val_acc_history = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(cust_model.state_dict())

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("_"*15)
        for phase in ["train", "valid"]:
            if phase == "train":
                cust_model.train()
            if phase == "valid":
                cust_model.eval()
            running_loss = 0.0
            jaccard_acc = 0.0
            dice_acc = 0.0

            for input_img, labels in dataloaders[phase]:
                input_img = input_img.cuda() if use_cuda else input_img
                labels = labels.cuda() if use_cuda else labels

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    outputs = cust_model(input_img)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input_img.size(0)
                jaccard_acc += jaccard(labels, preds)
                dice_acc += dice(labels, preds)

            epoch_loss = running_loss / len(dataloaders[phase])
            aver_jaccard = jaccard_acc / len(dataloaders[phase])
            aver_dice = dice_acc / len(dataloaders[phase])

            print("| {} Loss: {:.4f} | Jaccard Average Acc: {:.4f} | Dice Average Acc: {:.4f} |".format(phase, epoch_loss, aver_jaccard, aver_dice))
            if phase == "valid" and aver_jaccard > best_acc:
                best_acc = aver_jaccard
                best_model_wts = copy.deepcopy(cust_model.state_dict)
                pass
            if phase == "valid":
                val_acc_history.append(aver_jaccard)
                pass
        print("="*15)
        print(" ")
    time_elapsed = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best validation Accuracy: {:.4f}".format(best_acc))
    best_model_wts = copy.deepcopy(cust_model.state_dict())   # Need to change this in the future when I fix the jaccard index
    cust_model.load_state_dict(best_model_wts)
    return cust_model, val_acc_history

segm_model, acc = train_model(segm_model, dataloader_dict, criterion, optimizer, nr_epochs)
save_model(segm_model, name = "fcn_15epch.pt")

