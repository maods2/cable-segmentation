from pathlib import Path
from json import  loads
import numpy as np
from skimage.draw import polygon
from PIL import Image
import cv2
import matplotlib.pyplot as plt

path = './data_original_size/'
files = [Path(file).name for file in Path(path).glob('*.jpg')]
files = [file for file in files if 'mask' not in file]

with open('not_found_mask.txt', 'r') as f:
    files = [line.replace('\n','.jpg') for line in f.readlines()]



for file in files:
    try:
        original_image = Image.open(path+file).convert("RGB")
        file_name = file.split('.')[0]
        mask_data = loads(open(path + file_name + '.json').read())

        # print(mask_data['imageHeight'])
        # print(mask_data['imageWidth'])

        mask = np.zeros((mask_data['imageHeight'], mask_data['imageWidth']), dtype='uint8')

        x_max, y_max = original_image.size
        
        for shape in mask_data['shapes']:
            label = shape['label']
            if label == 'cable':
                x = [min(point[0], x_max-1) for point in shape['points']] 
                y = [min(point[1], y_max-1) for point in shape['points']] 

                x, y = np.array(x), np.array(y)
                rr, cc = polygon(y, x)

                mask[rr, cc] = 1

                
        binary_mask_8bit = (mask * 255).astype(np.uint8)
        

        # Convert to a Pillow Image
        mask_image = Image.fromarray(binary_mask_8bit)

        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)
        # plt.imshow(original_image)
        # plt.title("Imagem Original")

        # plt.subplot(1, 2, 2)
        # plt.imshow(mask_image, cmap='gray')
        # plt.title("Máscara Binária Gerada")

        # plt.show()
        

        mask_image.save(f'{path}{file_name}_mask.jpg')

    except Exception as e:
        print(e)

    