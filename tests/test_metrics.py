import sys

from utils import plot_loss_curve
sys.path.append('/workspaces/cable-segmentation/src')

import numpy as np
import matplotlib.pyplot as plt


    
metrics = np.load('/workspaces/cable-segmentation/checkpoints/model_metrics_20241110_235538.npz',allow_pickle=True).items()
metrics_dict = {key: value for key, value in metrics}
# print(metrics_dict)

plot_loss_curve(metrics_dict, 'loss_curve.png')