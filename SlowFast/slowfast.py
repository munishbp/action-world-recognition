import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")