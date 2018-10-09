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
batch_size = 8
nr_epocs = 10
momentum = 0.9
learning_rate = 0.001
running_loss = 0.0
gamma = 0.1
milestones = [1, 2, 3, 5, 7, 8]

segm_model = fcn_model(is_pretrained = True)
if use_cuda:
    segm_model.cuda()

transform = [ transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p = 1.0), 
             transforms.RandomRotation(205), transforms.RandomRotation(45), transforms.RandomRotation(145), 
             transforms.RandomRotation(300), transforms.ColorJitter(brightness=1.3), transforms.ColorJitter(contrast=1.2),
             transforms.ColorJitter(saturation=1.2), transforms.ColorJitter(saturation=0.7), transforms.ColorJitter(hue=0.3),
             transforms.ColorJitter(hue=0.1) ]

train_loader, validation_loader = Melanoma_Train_Validation_DataLoader(batch_size = batch_size, data_transforms = transforms.Compose([transform, transforms.ToTensor()]))

optimizer = optim.SGD(segm_model.parameters(), lr = learning_rate, momentum = momentum)
criterion = nn.BCEWithLogitsLoss().cuda() if use_cuda else nn.BCEWithLogitsLoss()

dataloader_dict = {"train": train_loader, "valid": validation_loader}

def train_model(cust_model, dataloaders, criterion, optimizer, num_epochs = 10, scheduler = None):
    start_time = time.time()
    val_acc_history = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epcoh {}/{}".format(epoch, num_epochs - 1))