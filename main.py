

import cv2
import matplotlib.pyplot as plt


def read_images(img):
    img = cv2.imread(img, 0)
    plt.imshow(img, cmap='gray')
    plt.show()
read_images('..//images//10.jpg')
read_images('..//images//8.jpg')