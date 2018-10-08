from data_loader import Melanoma_Train_Data
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import utils
from torchvision import transforms
from torch.utils.data import DataLoader

transforms_list = [ transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p = 1.0), 
                    transforms.RandomRotation(205), transforms.RandomRotation(45), transforms.RandomRotation(145), 
                    transforms.RandomRotation(300), transforms.ColorJitter(brightness=1.3), transforms.ColorJitter(contrast=1.2),
                    transforms.ColorJitter(saturation=1.2), transforms.ColorJitter(saturation=0.7), transforms.ColorJitter(hue=0.3),
                    transforms.ColorJitter(hue=0.1) ]


for idx, transform_choice in enumerate(transforms_list):
    train_data = Melanoma_Train_Data(data_transforms=transforms.Compose([transform_choice, transforms.ToTensor()]))
    dataloader = DataLoader(train_data, batch_size = 1, num_workers = 20, shuffle = False)

    for i, sample in enumerate(dataloader):
        img, label_img = sample
        utils.save_image(img, "ISIC_" + str("%07d" % ((16072*(idx+1))+i+1)) + ".jpg")
        utils.save_image(label_img, "ISIC_" + str("%07d" % ((16072*(idx+1))+i+1)) + "_segmentation.png")
    