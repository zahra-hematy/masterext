# import necessary library
import matplotlib.pyplot as plt


from sklearn.metrics import mean_squared_error
import cv2
import PIL
import glob
import numpy as np
from scipy.ndimage import rotate

# Load 20 Images Of Folder Twenty

image_list = []
binary_image_list_20 = []
for filename in glob.glob('.\\TwentyFingerprints\\*.BMP'):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # or im=Image.open(filename)
    image_list.append(im)
    # Convert 20 Image To Binary
    _, image_binary20 = cv2.threshold(im, 128, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    binary_image_list_20.append(image_binary20)

# Load 3 Image Of Folder One
image_List_one = []
binary_image_list_one = []
for FName in glob.glob('.\\One\\*.BMP'):
    IM = cv2.imread(FName, cv2.IMREAD_GRAYSCALE)
    image_List_one.append(IM)
    # Convert 3 Image To Binary
    _, image_binary = cv2.threshold(IM, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_image_list_one.append(image_binary)

# **************** for found which image needed rotate ***************
print(binary_image_list_one[0].shape)
print(binary_image_list_one[0][0])
print(binary_image_list_one[1].shape)
print(binary_image_list_one[2].shape)
print(binary_image_list_one[2][0])

# *********** rotate 2 images **********

v = binary_image_list_one[0]
b = rotate(v, -45)
b = b[50:148, 48:135]
b = cv2.resize(b, (96, 103))

cv2.imshow('ImageSlicing1', b)
cv2.waitKey()
vv = binary_image_list_one[1]
bb = rotate(vv, -62)
bb = bb[48:138, 48:135]
bb = cv2.resize(bb, (96, 103))

cv2.imshow('ImageSlicing2', bb)
cv2.waitKey()
plt.show()
bbb = binary_image_list_one[2]
L_COMP = [b, bb, bbb]

# ************** compare ********

x1 = []
x2 = []
x3 = []
for j in range(len(binary_image_list_20)):
    x1.append(mean_squared_error(L_COMP[0], binary_image_list_20[j]))
    x2.append(mean_squared_error(L_COMP[1], binary_image_list_20[j]))
    x3.append(mean_squared_error(L_COMP[2], binary_image_list_20[j]))
print(np.array(x1))
print(np.min(x1))
# ss = x1.sort()
print(np.sort(x1))

# print(np.array(x2))
# print(np.min(x2))
#
# print(np.array(x3))
# print(np.min(x3))











