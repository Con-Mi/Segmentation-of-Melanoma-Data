from data_loader import Melanoma_Train_Data
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms
from matplotlib import style
import pandas as pd

style.use("ggplot")

"""   ____ ALREADY DONE! ___
train_data = Melanoma_Train_Data(data_transforms=transforms.ToTensor())
dataloader = DataLoader(train_data, batch_size = 1, num_workers = 6, shuffle = True)
img_H = []
img_W = []

for sample in dataloader:
    img, _ = sample
    img_H.append(img.size(2))
    img_W.append(img.size(3))
H_W_dict = {"Height":img_H, "Width": img_W}
df = pd.DataFrame(H_W_dict)
df.to_csv("H_W_dims.csv")
"""
df = pd.read_csv("H_W_dims.csv")
print(df.head())
plt.scatter(df["Height"], df["Width"], alpha=0.2)
plt.show()