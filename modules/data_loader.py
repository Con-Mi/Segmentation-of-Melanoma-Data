import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class MelanomaDataset(Dataset):
    def __init__(self, file_list_data_idx, file_list_label_idx = None, transform = None, mode = "train"):
        self.data_root = "../data/"
        self.file_list_data_idx = file_list_data_idx
        self.file_list_label_idx = file_list_label_idx
        self.transform = transform
        self.mode = mode
        if self.mode is "train":
            self.data_dir = os.path.join(self.data_root, "ISIC2018_Task1-2_Training_Input/")
            self.label_dir = os.path.join(self.data_root, "ISIC2018_Task1_Training_GroundTruth/")
        elif self.mode is "test_valid":
            self.data_dir = os.path.join(self.data_root, "ISIC2018_Task1-2_Validation_Input/")
        elif self.mode is "final_test":
            self.data_dir = os.path.join(self.data_root, "ISIC2018_Task1-2_Test_Input/")

    def __len__(self):
        return len(self.file_list_data_idx)
    
    def __getitem__(self, index):
        if index not in range(len(self.file_list_data_idx)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list_data_idx["ids"].iloc[index]
        if self.mode is "train":
            train_id = self.file_list_data_idx["ids"].iloc[index]
            label_id = self.file_list_label_idx["ids"].iloc[index]
            self.image_path = os.path.join(self.data_dir, train_id)
            self.label_path = os.path.join(self.label_dir, label_id)
            image = Image.open(self.image_path)
            label = Image.open(self.label_path)
            if self.transform is not None:
                image = self.transform(image)
                label = self.transform(label)
            return image, label
        if self.mode is "test_valid":
            self.image_path = os.path.join(self.data_dir, file_id)
            image = Image.open(self.image_path)
            return image
        if self.mode is "final_test":
            self.image_path = os.path.join(self.data_dir, file_id)
            image = Image.open(self.image_path)
            return image

def Melanoma_Train_Data(data_transforms = None):
    file_list_input = pd.read_csv("./MelanomaData/train_input_ids.csv")
    file_list_label = pd.read_csv("./MelanomaData/train_labels_ids.csv")
    data_set = MelanomaDataset(file_list_input, file_list_label, transform = data_transforms, mode = "train")
    return data_set

def Melanoma_Train_Validation_DataLoader(batch_size = 4, validation_split = 0.1, num_workers = 8, data_transforms = None):
    file_list_input = pd.read_csv("../data/train_input_ids.csv")
    file_list_label = pd.read_csv("../data/train_labels_ids.csv")
    if data_transforms is None:
        data_transforms = transforms.ToTensor()
    dataset = MelanomaDataset(file_list_input, file_list_label, transform = data_transforms, mode = "train")

    shuffle_dataset = True
    random_seed = 1234

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler = train_sampler, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler = valid_sampler, num_workers=num_workers)
    
    return train_loader, validation_loader

def Melanoma_Test_Valid_Data(data_transforms = None):
    file_list_input = pd.read_csv("./MelanomaData/valid_input_ids.csv")
    data_set = MelanomaDataset(file_list_input, file_list_label_idx = None, transform = data_transforms, mode = "test_valid")
    return data_set

def Melanoma_Test_Data(data_transforms = None):
    file_list_input = pd.read_csv("./MelanomaData/test_input_ids.csv")
    data_set = MelanomaDataset(file_list_input, file_list_label_idx = None, transform = data_transforms, mode = "test")
    return data_set

