import torch
import numpy as np
from torchvision import transforms
import pandas as pd
import gc

from linknet import linknet_model
from data_loader import Melanoma_Train_Validation_DataLoader
from helper import jaccard, dice, load_model

use_cuda = torch.cuda.is_available()

# Hyperparameters
thrs_list = np.linspace(0.1, 0.9, 400) 
batch_size = 4
num_workers = 10

_, validation_loader = Melanoma_Train_Validation_DataLoader(batch_size = batch_size,  data_transforms=transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()]), num_workers=num_workers)

model = linknet_model().cuda() if use_cuda else linknet_model()
model = load_model(model, model_dir="linknet_10epch.pt", map_location_device="gpu") if use_cuda else load_model(model, model_dir="linknet_10epch.pt")
columns = ["Threshold", "Accuracy"]
thrs_df = pd.DataFrame(columns = columns)
thrs_df["Threshold"] = thrs_list
del columns

for thrs in thrs_list:
    jaccard_acc = 0.0
    for input_img, label_img in validation_loader:
        input_img = input_img.cuda() if use_cuda else input_img
        label_img = label_img.cuda() if use_cuda else label_img
        outputs = model(input_img)
        preds = torch.sigmoid(outputs)
        jaccard_acc += jaccard(label_img, (preds > thrs).float())
    print("Threshold {:.8f} | Jaccard Accuracy: {:.8f}".format(thrs, jaccard_acc / len(validation_loader)))
    thrs_df["Accuracy"] = (jaccard_acc / len(validation_loader))
    gc.collect()

idx = thrs_df.loc[thrs_df["Accuracy"] == max(thrs_df["Accuracy"])]
optimal_thrs = thrs_df["Threshold"].loc[idx.index.values]
thrs_df.to_csv("accuracyVSthreshold.csv")
print("Optimal Threshold is {:.8f} found with Accuracy of {:.4f}".format(optimal_thrs.values, max(thrs_df["Accuracy"])))
