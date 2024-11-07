from pathlib import Path
from json import  loads
import numpy as np
from skimage.draw import polygon
from PIL import Image
import cv2
import matplotlib.pyplot as plt

path = './data_original_size/'
masks = [Path(file).name for file in Path(path).glob('*mask.jpg')]
book = {}

for file in masks:
    book[Path(file).name.split('.')[0].replace('_mask','')] = 0


files = [Path(file).name for file in Path(path).glob('*.jpg') if 'mask' not in Path(file).name]
not_found = []
couter = 0
for file in files:
    key = Path(file).name.split('.')[0]
    if book.get(key) is None:
        print('not found', key)
        couter +=1
        not_found.append(key)
        
# print(not_found)
# print(book)

with open('./not_found_mask.txt', 'w') as f:
    for i in not_found:
        f.write(i + '\n')



