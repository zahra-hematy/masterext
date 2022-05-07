import cv2
import numpy as np
from PIL import Image
import  pandas as pd
from PIL import Image
import PIL
from sklearn import preprocessing
import  scipy

def colorizeBy_colorPalette(image_addr: str, colorPalette_addr: str, output_addr: str = "."):
    """
    Colorizes an image using a color palette.

    [Args]:
        image_addr: str. The address of the gray image to be colorized.
        colorPalette_addr: str. The address of the color palette. CSV file with 256 color and [R, G, B] in columns.

    [Returns]:
        None. It saves automatically the colorized image.
    """
    grayimg = Image.open(image_addr)
    grayimg_arr = np.array(grayimg)
    colors_data = pd.read_csv(colorPalette_addr)
    R = np.zeros(grayimg_arr.shape)
    G = np.zeros(grayimg_arr.shape)
    B = np.zeros(grayimg_arr.shape)
    for col in range(grayimg_arr.shape[1]):
        for row in range(grayimg_arr.shape[0]):
            R[row, col] = colors_data.iloc[grayimg_arr[row, col], 0]
            G[row, col] = colors_data.iloc[grayimg_arr[row, col], 1]
            B[row, col] = colors_data.iloc[grayimg_arr[row, col], 2]
    out = np.array((R, G, B)).transpose(1, 2, 0)
    new = Image.fromarray(out.astype('uint8'), 'RGB')
    new.save(f'{output_addr}/Image_color.jpg')

# Read image gray 1
x = cv2.imread('.\\Forest_1/Forest_1_S_gray.jpg ')
# Read image gray 2
#x = cv2.imread('.\\Forest_2/Forest_2_S_gray.jpg ')
# Read image gray 3
#x = cv2.imread('.\\Forest_3/Forest_3_S_gray.jpg ')

# show image gray
cv2.imshow('ImagesGray', x)
cv2.waitKey(1)


# send image and csv for function
#b = colorizeBy_colorPalette('.\\Forest_1/Forest_1_S_gray.jpg','.\\Forest_1/Forest_1_S_color_palette_sort.csv',".")
#b = colorizeBy_colorPalette('.\\Forest_2/Forest_2_S_gray.jpg', '.\\Forest_2/Forest_2_S_color_palette_sort.csv  ',".")
b = colorizeBy_colorPalette('.\\Forest_3/Forest_3_S_gray.jpg', '.\\Forest_3/Forest_3_S_color_palette_sort.csv  ',".")

# calculate difference index image and image gray
x = np.array([x])
bn = np.array([b])
for col in range(x.shape[1]):
    for row in range(bn.shape[0]):
        er = col - row
print(er)