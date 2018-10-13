import torch
import numpy as np
from torchvision import transforms

from fcn_naive_model import fcn_model
from data_loader import Melanoma_Train_Validation_DataLoader
from helper import jaccard, dice, load_model

use_cuda = torch.cuda.is_available()

# Hyperparameters
thrs_list = np.linspace(0.5, 0.95, 200) 
batch_size = 2
num_workers = 4

_, validation_loader = Melanoma_Train_Validation_DataLoader(batch_size = batch_size,  data_transforms=transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()]), num_workers=0)

model = fcn_model().cuda() if use_cuda else fcn_model()
model = load_model(model, model_dir="fcn_15epch_interpol.pt", map_location_device="gpu") if use_cuda else load_model(model, model_dir="fcn_15epch_interpol.pt")
thrs_dict = {}

for thrs in thrs_list:
    jaccard_acc = 0.0
    for input_img, label_img in validation_loader:
        input_img = input_img.cuda() if use_cuda else input_img
        label_img = label_img.cuda() if use_cuda else label_img
        outputs = model(input_img)
        preds = torch.sigmoid(outputs)
        jaccard_acc += jaccard(label_img, (preds > thrs).float())
    print("Threshold {:.8f} | Jaccard Accuracy: {:.8f}".format(thrs, jaccard_acc / len(validation_loader)))
    thrs_dict[thrs] = (jaccard_acc / len(validation_loader))