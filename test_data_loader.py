import torch

from data_loader import Melanoma_Train_Validation_DataLoader
from torchvision import transforms
from fcn_naive_model import fcn_model

from helper import jaccard, dice

use_cuda = torch.cuda.is_available()
segm_model = fcn_model()
train_loader, validation_loader = Melanoma_Train_Validation_DataLoader(batch_size = 4, data_transforms=transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()]), num_workers=2)
if use_cuda:
    segm_model.cuda()
segm_model.train()

for i, sample in enumerate(validation_loader):
    img, label_img = sample
    img = img.cuda() if use_cuda else img
    label_img = label_img.cuda() if use_cuda else label_img
    output = segm_model(img)
    out = torch.sigmoid(output)
    print("The Jaccard accuracy is: {:.4f}".format(jaccard(label_img, out)))
    print("The Dice accuracy is: {:.4f}".format(dice(label_img, out)))


