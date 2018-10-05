from data_loader import Melanoma_Train_Data
from torch.utils.data import DataLoader
from torchvision import transforms as Trf

train_data_set = Melanoma_Train_Data(data_transforms = Trf.Compose([Trf.Resize([512, 512]), Trf.ToTensor()]))
train_dataloader = DataLoader(train_data_set, batch_size = 4, shuffle = True, num_workers=6)

for i, sample in enumerate(train_dataloader):
    input_img, img_label = sample
    print(input_img.size())
    print(i)

