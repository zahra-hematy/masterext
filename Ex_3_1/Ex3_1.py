import  cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image

# read Images
path = '.\\Face_gray.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Extract bit pages and give value to each page
out = []
for k in range(0 , 8):
    plane = np.full((img.shape[0], img.shape[1]),
                    2**k, dtype=np.uint8)
    res = cv2.bitwise_and(plane, img)
    x = res * 255
    out.append(x)
    # cv2.imshow(f"bit plane {k}",np.array(out[i]))
out[0 ] = cv2.normalize( out[0].astype('float') ,None, 0.0,1.0,cv2.NORM_MINMAX)
out[1] = cv2.normalize( out[1].astype('float') ,None, 0.0,2.0,cv2.NORM_MINMAX)
out[2 ] = cv2.normalize( out[2].astype('float') ,None, 0.0,4.0,cv2.NORM_MINMAX)
out[3] = cv2.normalize( out[3].astype('float') ,None, 0.0,8.0,cv2.NORM_MINMAX)
out[4] = cv2.normalize( out[4].astype('float') ,None, 0.0,16.0,cv2.NORM_MINMAX)
out[5] = cv2.normalize( out[5].astype('float') ,None, 0.0,32.0,cv2.NORM_MINMAX)
out[6 ] = cv2.normalize( out[6].astype('float') ,None, 0.0,64.0,cv2.NORM_MINMAX)
out[7 ] = cv2.normalize( out[7].astype('float') ,None, 0.0,128.0,cv2.NORM_MINMAX)

# read and resize to 96*100 and convert to binary
thumb = cv2.imread('.\\11.BMP' , 0)
thumb = cv2.resize(thumb, (100,96))
_, finger = cv2.threshold(thumb, 125, 255, cv2.THRESH_BINARY)
plt.imshow(finger, cmap='gray')
plt.show()
finger = finger /255
print(f"finger.shape: {finger.shape}")

# put finger to image with reshape and slicing
finger_shape = finger.shape
finger = finger.reshape(-1)
b0_shape = out[0].shape
buffer = out[0].copy().reshape(-1)
buffer[:9600] = finger
out[0] = buffer.reshape(b0_shape)

# show bit 0
plt.figure(2)
plt.imshow(out[0], cmap='gray')
plt.show()

# recovery main image
OriginalImage = np.zeros((640, 640), dtype=np.uint8)
for i in range(OriginalImage.shape[0]):
    for j in range(OriginalImage.shape[1]):
        for data in range(8):
            x = np.array([OriginalImage[i, j]], dtype=np.uint8)
            data = np.array([data], dtype=np.uint8)
            flag = np.array([0 if out[data[0]][i, j] == 0 else 1], dtype=np.uint8)
            mask = flag << data[0]
            x[0] = (x[0] & ~mask) | ((flag[0] << data[0]) & mask)
            OriginalImage[i, j] = x[0]

# show Original image (encrypted)
plt.figure(10)
plt.imshow(OriginalImage, cmap='gray')
plt.show()
# save image encrypted
cv2.imwrite("comb_Finger_line_encrypt.png", OriginalImage)


