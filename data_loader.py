from torch.utils import data
from torch.utils.data import dataset
from torchvision import utils
from torchvision import transforms as Trf
import os
import pandas as pd
from PIL import Image
import numpy as np

class MelanomaDataset(data.Dataset):
    def __init__(self, file_list_idx, transform = None, mode = "train"):
        self.data_root = "./MelanomaData"
        self.file_list_idx = file_list_idx
        self.transform = transform
        self.mode = mode
        if self.mode is "train":
            self.data_dir = os.path.join(self.data_root, "ISIC2018_Task1_Training_Input/")
            self.label_dir = os.path.join(self.data_root, "ISIC2018_Task1_Training_GroundTruth")
        elif self.mode is "valid":
            self.data_dir = os.path.join(self.data_root, "") # Note a 10% of the training data to be used as validation data
            self.label_dir = os.path.join(self.data_root, "")
        elif self.mode is "test_valid":
            self.data_dir = os.path.join(self.data_root, "ISIC2018_Task1-2_Validation_Input/")
        elif self.mode is "final_test":
            self.data_dir = os.path.join(self.data_root, "ISIC2018_Task1-2_Test_Input/")

    def __len__(self):
        return len(self.file_list_idx)
    
    def __getitem__(self, index):
        if index not in range(len(self.file_list_idx)):
            return self.__getitem__(np.random.randint(o, self.__len__()))

        file_id = self.file_list_idx["ids"].iloc[index]

        if self.mode is "train":
            self.image_path = os.path.join(self.data_dir, file_id)
            self.label_path = os.path.join(self.label_dir, file_id)
            image = Image.open(self.image_path)
            label = Image.open(self.label_path)
            if self.transform is not None:
                image = self.transform(image)
                label = self.transform(label)
            return image, label
        if self.mode is "valid":
            self.image_path = os.path.join(self.data_dir, file_id)
            self.label_path = os.path.join(self.label_dir, file_id)
            image = Image.open(self.image_path)
            label = Image.open(self.label_path)
            if self.transform is not None:
                image = self.transform(image)
                label = self.transform(label)
            return image, label
        if self.mode is "test_valid":
            self.image_path = os.path.join(self.data_dir, file_id))
            image = Image.open(self.image_path)
            return image
        if self.mode is "final_test":
            self.image_path = os.path.join(self.data_dir, file_id)
            image = Image.open(self.image_path)
            return image

def Melanoma_Train_Data(data_transforms = None):
    file_list = pd.read_csv()
    data_set = MelanomaDataset(file_list, transform = data_transforms, mode = "train")
    return data_set

def Melanoma_Valid_Data(data_transforms = None):
    file_list = pd.read_csv()
    data_set = MelanomaDataset(file_list, transform = data_transforms, mode = "valid")
    return data_set

def Melanoma_Test_Valid_Data(data_transforms = None):
    file_list = pd.read_csv()
    data_set = MelanomaDataset(file_list, transform = data_transforms, mode = "test_valid")
    return data_set

def Melanoma_Test_Data(data_transforms = None):
    file_list = pd.read_csv()
    data_set = MelanomaDataset(file_list, transform = data_transforms, mode = "test")
    return data_set

