import os
from pathlib import Path
import torch


def save_model(model):
    Path(f'./checkpoints/').mkdir(exist_ok=True)
    model_path = f"./checkpoints/model.pth"
    torch.save(model.state_dict(), model_path)
    
def load_checkpoint(path, model):
    print("=> Loading checkpoint")
    model.load_state_dict(torch.load(path))