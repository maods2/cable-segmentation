import sys

sys.path.append('/workspaces/cable-segmentation/src')
from utils import plot_loss_curve

import numpy as np
import matplotlib.pyplot as plt


path = '/workspaces/cable-segmentation/checkpoints/model_metrics_20241111_005256.npz'
metrics = np.load(path,allow_pickle=True).items()
metrics_dict = {key: value for key, value in metrics}
# print(metrics_dict)

plot_loss_curve(metrics_dict, 'loss_curve.png')