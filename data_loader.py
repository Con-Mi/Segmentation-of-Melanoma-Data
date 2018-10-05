from torch.utils import data
from torch.utils.data import dataset
from torchvision import utils
from torchvision import transforms as Trf
import os
import pandas as pd
from PIL import Image
import numpy as np

class MelanomaDataset(data.Dataset):
    