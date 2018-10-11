from data_loader import Melanoma_Train_Validation_DataLoader
from torchvision import transforms
from fcn_naive_model import fcn_model
import torch

use_cuda = torch.cuda.is_available()
segm_model = fcn_model()
train_loader, validation_loader = Melanoma_Train_Validation_DataLoader(batch_size = 4, data_transforms=transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()]))
if use_cuda:
    segm_model.cuda()
segm_model.train()

for i, sample in enumerate(validation_loader):
    img, label_img = sample
    output = segm_model(img)
    print(output.size())
    print(label_img.size())
    print(type(img))

