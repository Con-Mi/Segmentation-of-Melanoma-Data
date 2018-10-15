import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision import utils

from fcn_naive_model import fcn_model
from helper import load_model, mask_overlay

import time
import cv2
import numpy as np

start_time = time.time()
segm_model = fcn_model()
segm_model = load_model(segm_model, model_dir="fcn_15epch_interpol.pt")
# img = Image.open("sample2.jpg")
img = Image.open("ISIC_0012255.jpg")
trf = transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()])
img_input = trf(img)
img_input = img_input.unsqueeze(dim=0)
output = segm_model(img_input)
output = torch.sigmoid(output)

inference_time = time.time() - start_time
print("Time to do inference was: {:.0f} seconds and {:.5f} micro seconds ".format(inference_time, inference_time % 60))
output = output.squeeze(0)
output = output.permute(1, 2, 0)
output = output.squeeze()
output_np = output.detach().numpy()
thrs = 0.37090301
upper = 1
lower = 0
binary_output = np.where(output_np > thrs, upper, lower)
# plt.imsave("bitmap_sample.png", binary_output, cmap = "tab20b")
plt.imsave("bitmap_melanoma.png", binary_output, cmap = "tab20b")

save_img = img_input.squeeze(0)
save_img = save_img.permute(1, 2, 0)
save_img = save_img.squeeze()
save_img = save_img.detach().numpy()
# plt.imsave("original_sample.png", save_img)
plt.imsave("original_melanoma_sample.png", save_img)

# orig_img = cv2.imread("original_sample.png")
# mask = cv2.imread("bitmap_sample.png")
orig_img = cv2.imread("original_melanoma_sample.png")
mask = cv2.imread("bitmap_melanoma.png")

overlayed_img = cv2.addWeighted(orig_img, 1, mask, 0.3, 0, 0)
# cv2.imwrite("overlayed_img.png", overlayed_img)
cv2.imwrite("overlayed_malanome_img.png", overlayed_img)

end_program_time = time.time() - start_time
print("Total time to run inference and save the image: {:.0f} seconds and {:.5f} micro seconds ".format(end_program_time, end_program_time % 60))