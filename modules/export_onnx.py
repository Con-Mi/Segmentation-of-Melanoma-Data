from torch import randn
from torch import onnx

from linknet_batch import linknet_batch_model
from helper import load_model

exp_model = linknet_batch_model()
exp_model = load_model(exp_model, model_dir="linknet_10epch.pt")

dummy_input = randn(1, 3, 512, 512)

onnx.export(exp_model, dummy_input, "linknet.onnx")