import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
#import seaborn as sns
from torchvision import utils
import numpy as np

from fcn_naive_model import fcn_model
from helper import load_model, mask_overlay

from matplotlib import style
import time

style.use("ggplot")

start_time = time.time()
segm_model = fcn_model()
segm_model = load_model(segm_model, model_dir="fcn_15epch_interpol.pt")
img = Image.open("sample2.jpg")
trf = transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()])
img_input = trf(img)
img_input = img_input.unsqueeze(dim=0)
output = segm_model(img_input)
output = torch.sigmoid(output)

inference_time = time.time() - start_time
print("Time to do inference was: {:.0f} seconds and {:.0f} micro seconds ".format(inference_time, inference_time % 60))
output = output.squeeze(0)
output = output.permute(1, 2, 0)
output = output.squeeze()
output_np = output.detach().numpy()
thrs = 0.7
upper = 1
lower = 0
binary_output = np.where(output_np > thrs, upper, lower)
#plt.imshow(binary_output)
#plt.show()
plt.imsave("bitmap_sample.png", binary_output)

save_img = img_input.squeeze(0)
save_img = save_img.permute(1, 2, 0)
save_img = save_img.squeeze()
save_img = save_img.detach().numpy()
plt.imsave("original_sample.png", save_img)
end_program_time = time.time() - start_time
print("Total time to run inference and save the image: {:.0f} seconds ".format(end_program_time))
"""
f, ax = plt.subplots(1, 3)
sns.heatmap(output_np, ax = ax[0])
sns.heatmap(binary_output, ax = ax[1])
ax[2].imshow(img)
plt.show()
"""
img = img.resize((512, 512))
img = np.asarray(img)
print("Image shape is: ")
print(img.shape)
print("Binary output shape is: ")
binary_output = np.resize(binary_output, (512, 512, 1))
print(binary_output.shape)
print("Type of image")
print(type(img))
print("Type of binary")
print(binary_output)

# ERROR HERE. ERROR WITH OPENCV.
overlay_img = mask_overlay(img, binary_output)
plt.imshow(overlay_img)
plt.show()