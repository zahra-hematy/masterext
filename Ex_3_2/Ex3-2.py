import  cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import mean_squared_error
import glob

# read image encrypted
path = '.\\Comb_Finger_line.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Extract bit pages and give value to each page
out = []
for k in range(0 , 8):
    plane = np.full((img.shape[0], img.shape[1]),
                    2**k, dtype=np.uint8)
    res = cv2.bitwise_and(plane, img)
    x = res * 255
    out.append(x)

out[0 ] = cv2.normalize( out[0].astype('float') ,None, 0.0,1.0,cv2.NORM_MINMAX)
out[1] = cv2.normalize( out[1].astype('float') ,None, 0.0,2.0,cv2.NORM_MINMAX)
out[2 ] = cv2.normalize( out[2].astype('float') ,None, 0.0,4.0,cv2.NORM_MINMAX)
out[3] = cv2.normalize( out[3].astype('float') ,None, 0.0,8.0,cv2.NORM_MINMAX)
out[4] = cv2.normalize( out[4].astype('float') ,None, 0.0,16.0,cv2.NORM_MINMAX)
out[5] = cv2.normalize( out[5].astype('float') ,None, 0.0,32.0,cv2.NORM_MINMAX)
out[6 ] = cv2.normalize( out[6].astype('float') ,None, 0.0,64.0,cv2.NORM_MINMAX)
out[7 ] = cv2.normalize( out[7].astype('float') ,None, 0.0,128.0,cv2.NORM_MINMAX)
out = np.array(out)

# show each page
cv2.imshow('0' , out[0])
cv2.waitKey(1)
print('0 : \n ',out[0])
cv2.imshow('1' , out[1])
cv2.waitKey(1)
print('1 : \n ',out[1])
cv2.imshow('2' , out[2])
cv2.waitKey(1)
print('2 : \n ',out[2])
cv2.imshow('3' , out[3])
cv2.waitKey(3)
print('3 : \n ',out[0])
cv2.imshow('4' , out[4])
cv2.waitKey()
print('4 : \n ',out[4])
cv2.imshow('5' , out[5])
cv2.waitKey()
print('5 : \n ',out[5])
cv2.imshow('6' , out[6])
cv2.waitKey()
print('6 : \n ',out[6])
cv2.imshow( '7 : \t ',out[7])
cv2.waitKey()
print('7 : \n ',out[7])


plt.figure(1)
plt.subplot(2, 4, 1)
# Show every bit plane as separated images
for i in range(8):
    plt.subplot(2, 4, i + 1)
    # Convert [0 - 255] to [0.0 - 1.0]
    ss = cv2.normalize(out[i].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imshow(np.array(ss), cmap='gray')
plt.show()

# read finger with 96*103 for use its shape and decrypt
path2 = '.\\11.BMP '
path1 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

# convert to binary
_, finger = cv2.threshold(path1, 128, 255, cv2.THRESH_BINARY)
# [0,1]
finger = finger /255
# convert to array fo use shape
finger = np.array(finger)
finger_shape = finger.shape
# Linearize the image
z = out[0].copy().reshape(-1,)
# convert to binarry for use slicing

z = np.array(z)
ext_finger = z[:9888]
print(f"z shape:{z.shape}")
plt.imshow(np.array(ext_finger).reshape(finger_shape), cmap='gray')
plt.show()
ext_finger = (np.array(ext_finger).reshape(finger_shape))
#cv2.imwrite('finger_decrypt.png', ext_finger)

# # compare with 20 images
x1 = []
image_list = []
binary_image_list_20 = []
for filename in glob.glob('.\\TwentyFingerprints\\*.BMP'):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # or im=Image.open(filename)

    image_list.append(im)
    # Convert 20 Image To Binary
    _, image_binary20 = cv2.threshold(im, 128, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    binary_image_list_20.append(image_binary20)

_, ext_finger = cv2.threshold(ext_finger, 128, 255, cv2.THRESH_BINARY)
ext_finger = np.array(ext_finger)
binary_image_list_20 = np.array(binary_image_list_20)
for j in range(20):
    x1.append(mean_squared_error(ext_finger, binary_image_list_20[j]))
print(x1)
print(ext_finger.shape)
print(binary_image_list_20.shape)
