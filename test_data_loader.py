from data_loader import Melanoma_Train_Validation_DataLoader
from torchvision import transforms

train_loader, validation_loader = Melanoma_Train_Validation_DataLoader(batch_size = 1, data_transforms=transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()]))

for i, sample in enumerate(validation_loader):
    img, label_img = sample
    print(img.size())
    print(label_img.size())
    print(type(img))
