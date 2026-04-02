import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import fvcore
import json
from torch.utils.data import DataLoader, Dataset

# Other models
# _slowfast
# slowfast_r50
# slowfast_r101
# slowfast_16x8_r101_50_50
# slowfast_r50_detection




class SlowFastPathway(nn.Module):
    def __init__():
        super().__init__()
    

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


transform = transforms.Compose([])
loss_function = nn.BCEWithLogitsLoss()

def trainingModel(model, learning_rate, optimizer, epochs):
    #model.train()
    pass


def evalationModel(model, learning_rate, optimizer):
    pass
    #with torch.no_grad():
        #layer = model[0]
        #weights = layer.weight.data



def main():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crop_size = 240
    frames_per_second = 30
    slowfast_alpha = 3
    epochs = 25
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)