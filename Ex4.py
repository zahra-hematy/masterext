# import necessary library
import glob
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.filters import median
from skimage.filters import gaussian
import image
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from skimage.feature import canny
from scipy.ndimage import distance_transform_edt


# add noise salt and pepper
def salt_pepper(img):
    # Getting the dimensions of the image
    row, col = img.shape
    # Randomly pick some pixels in the
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)
        # Color that pixel to white (salt)
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)
        # Color that pixel to black (pepper)
        img[y_coord][x_coord] = 0
    return img
# Reading the color image in grayscale image
img = cv2.imread('.//EX4-DataSet//Original/1.jpg', cv2.IMREAD_GRAYSCALE)
# Storing the image
cv2.imwrite('.//EX4-DataSet//Noised//salt-pepper.jpg', salt_pepper(img))

# ------------>  second way to creat noise s-p image

#Reading the color image in grayscale image
img1 = cv2.imread('.//EX4-DataSet//Original/1.jpg', cv2.IMREAD_GRAYSCALE)
# use skimage with mode s and p for add noise
img_noise_sp = skimage.util.random_noise(img1, mode='s&p', amount=0.3)       # -->default 0.5
# use skimage for creat filter median with win 3*3
median_filter = skimage.filters.median(img_noise_sp, skimage.morphology.disk(3))
plt.subplot(131)
# show image original
plt.imshow(img1, cmap='gray')
plt.title("without noise")
plt.subplot(132)
# show image with noise s & p
plt.imshow(img_noise_sp, cmap='gray')
plt.title("with noise s&p")
# show Improved image with filter median
plt.subplot(133)
plt.imshow(median_filter, cmap='gray')
plt.title("Improved filter median")
plt.show()
cv2.imwrite('.//EX4-DataSet//Noise_eliminated//remove_salt-pepper.jpg', median_filter)


# add noise addGaussianNoise
def addGaussianNoise(src):
    # get shape (x , y)
    row, col, ch = src.shape
    # define parameter
    mean = 0
    sigma = 15
    # distribution normal with mean -> 0, sigma -> 15
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    # reshape to shape image
    gauss = gauss.reshape(row, col, ch)
    # concatenation
    noisy = src + gauss
    return noisy
# Reading the color image in grayscale image
src = cv2.imread('.//EX4-DataSet//Original/2.jpg', None)
# Storing the image
cv2.imwrite('.//EX4-DataSet//Noised//addGaussianNoise.jpg', addGaussianNoise(src))

# second way to creat noise guassian image
# Reading the color image in grayscale image
img2 = cv2.imread('.//EX4-DataSet/Original/2.jpg', 0)
# use skimage with mode guassian and varyance 0.04(default) & mean 0 for add noise guassian
img_noise_gaussian = skimage.util.random_noise(img2, 'gaussian', mean=0, var=0.03)
# Use skimage for creat filter gaussian
gaussian_filter = skimage.filters.gaussian(img_noise_gaussian, sigma=3)
plt.subplot(131)
# show image original
plt.imshow(img2, cmap='gray')
plt.title("Image without noise")
plt.subplot(132)
# show image with noise guassian
plt.imshow(img_noise_gaussian, cmap='gray')
plt.title("noise gaussian")
plt.subplot(133)
plt.imshow(gaussian_filter, cmap='gray')
plt.title("Improved with filter gaussian")
plt.show()
cv2.imwrite('.//EX4-DataSet//Noise_eliminated//remove_gaussian.jpg', gaussian_filter)

# add noise periodic_noise
# Read image anc normalized it between 0.0 and 1.0 as float type
img3 = cv2.normalize(cv2.imread('.//EX4-DataSet/Original/3.jpg', cv2.IMREAD_GRAYSCALE).astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Create periodic noise
# row and column image
[m, n] = img3.shape
# to create a rectangular grid out of an array of n values and an array of m values.
[a, b] = np.meshgrid(np.arange(0, n), np.arange(0, m))
p = np.sin(a / 2 + b / 2) + 1
# Adding made noise to the image
img_noise_periodic = cv2.normalize((img3 + p) / 2, None, 0, 255, cv2.NORM_MINMAX)
plt.subplot(121)
# show image original
plt.imshow(img3, cmap='gray')
plt.title("Image without noise")
plt.subplot(122)
# show image with noise periodic
plt.imshow(img_noise_periodic, cmap='gray')
plt.title("Image with noise periodic")
plt.show()
# Storing the image
cv2.imwrite('.//EX4-DataSet//Noised//periodic_noise.jpg', img_noise_periodic)
# remove noise for noise_periodic
plt.figure(1)
# Fourier transform image
imgF1 = np.fft.fftshift(np.fft.fft2(img3))
imgFd1 = np.abs(imgF1)
imgFdL1 = np.log(imgFd1)
plt.imshow(imgFdL1, cmap="gray")
plt.show()
Max1 = np.max(np.max(imgFdL1))
# Apply Fourier transform to noisy image
imgnF2 = np.fft.fftshift(np.fft.fft2(img_noise_periodic))
imgnFD2 = np.abs(imgnF2)
imgnFDL2 = np.log(imgnFD2)
Max2 = np.max(np.max(imgnFDL2))
imgSF2 = imgnFDL2 / Max2
# Remove noise in fourier series with notch filter
imgnF2[146:152, :] = 0
imgnF2[206:212, :] = 0
imgnF2[:, 364:370] = 0
imgnF2[:, 268:272] = 0
NoisyImgFd = np.abs(imgnF2)
Max = np.max(np.max(imgnF2))
IMGNotchLog = np.log(1 + NoisyImgFd)
MaxNotch = np.max(np.max(IMGNotchLog))
plt.subplot(141)
# show image original
plt.imshow(img3, cmap='gray')
plt.title("without noise")
plt.subplot(142)
# show image with noise periodic
plt.imshow(img_noise_periodic, cmap='gray')
plt.title("noise periodic")
plt.subplot(143)
plt.imshow(IMGNotchLog / MaxNotch, cmap='gray', vmin=0, vmax=1)
plt.title("Notch mask")
# inverse fft
notchImgInv = np.abs(np.fft.ifft2(imgnF2))
plt.subplot(144)
plt.imshow(notchImgInv / np.max(np.max(notchImgInv)), cmap='gray', vmin=0, vmax=1)
n =notchImgInv / np.max(np.max(notchImgInv))
n = n*255
plt.title("Reducing noise")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()
# save image improve
cv2.imwrite('.//EX4-DataSet//Noise_eliminated//remove_periodic_noise.jpg', notchImgInv)

# add noise speckle
img4 = cv2.imread('.//EX4-DataSet/Original/4.jpg', 0)
# use skimage with mode speckle and varyance 0.7 & mean 0
img_noise_speckle = skimage.util.random_noise(img4, 'speckle', mean=0, var=0.07)
plt.subplot(121)
# show image original
plt.imshow(img4, cmap='gray')
plt.title("Image without noise")
plt.subplot(122)
# show image with noise speckle image
plt.imshow(img_noise_speckle, cmap='gray')
plt.title("Image with noise speckle")
plt.show()
# convert to [0,255] for save
i = img_noise_speckle*255
cv2.imwrite('.//EX4-DataSet/Noised/noise_speckle.jpg', i)
# Use skimage for creat filter gaussian
gaussian_filter1 = skimage.filters.gaussian(img_noise_speckle, sigma= 1)
plt.subplot(131)
# show image original
plt.imshow(img4, cmap='gray')
plt.title("Image without noise")
plt.subplot(132)
# show image with noise guassian
plt.imshow(img_noise_speckle, cmap='gray')
plt.title("noise speckle")
plt.subplot(133)
plt.imshow(gaussian_filter1, cmap='gray')
plt.title("Improved with filter gaussian")
plt.show()
# convert to [0,255] for save
im = gaussian_filter1*255
#save
cv2.imwrite('.//EX4-DataSet/Noise_eliminated/remove_speckle.jpg', im)
# second way for delete noise speckle
def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = variance(img)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output
# show improved image with lee
im = lee_filter(img_noise_speckle, 10)
# convert to [0,255] for save
im = im*255
#save
cv2.imwrite('.//EX4-DataSet/Noise_eliminated/remove_speckle.jpg', im)
# show
plt.imshow(lee_filter(img_noise_speckle, 12), cmap='gray')
plt.title("Reducing noise speckle")
plt.show()

#---------------->>>>>>>>>> second part (b)

def MSE(path1, path2):
    # Read Original image --> this image doesn't have noise
    im1 = np.array(cv2.imread(path1, cv2.IMREAD_GRAYSCALE))
    # Read Noisy image
    im2 = np.array(cv2.imread(path2, cv2.IMREAD_GRAYSCALE))
    # Find image shape
    [M, N] = im1.shape
    # Calc Mean Square Error
    MSE = (np.sum(np.sum((im2 - im1) ** 2))) / (M * N)
    print(np.array(MSE))
# MSE( './/EX4-DataSet//Original//1.jpg', './/EX4-DataSet//Noise_eliminated//remove_salt-pepper.jpg')   #---------->108.16332870911215
# MSE( './/EX4-DataSet//Original//2.jpg', './/EX4-DataSet//Noise_eliminated//remove_gaussian.jpg')  #---------->114.24928310740354
# MSE('.//EX4-DataSet//Original//3.jpg', './/EX4-DataSet//Noise_eliminated//remove_periodic_noise.jpg')  #---------->105.31945117728532
# MSE('.//EX4-DataSet//Original//4.jpg', './/EX4-DataSet//Noise_eliminated//remove_speckle.jpg')  #---------->56.18701697892271


def MAE(path1, path2):
    # Read Original image --> this image doesn't have noise
    im1 = np.array(cv2.imread(path1, cv2.IMREAD_GRAYSCALE))
    # Read Noisy image
    im2 = np.array(cv2.imread(path2, cv2.IMREAD_GRAYSCALE))
    # Find image shape
    [M, N] = im1.shape
    # Calc Mean Absolute Error
    MAE = (np.sum(np.sum(np.abs(im2 - im1)))) / (M * N)
    print(np.array(MAE))
# MAE( './/EX4-DataSet//Original//1.jpg', './/EX4-DataSet//Noise_eliminated//remove_salt-pepper.jpg')   #---------->95.04635294976636
# MAE( './/EX4-DataSet//Original//2.jpg', './/EX4-DataSet//Noise_eliminated//remove_gaussian.jpg')  #---------->85.97791482012514
# MAE('.//EX4-DataSet//Original//3.jpg', './/EX4-DataSet//Noise_eliminated//remove_periodic_noise.jpg')  #---------->
# MAE('.//EX4-DataSet//Original//4.jpg', './/EX4-DataSet//Noise_eliminated//remove_speckle.jpg')  #---------->

def WCAE(path1, path2):
    # Read Original image --> this image doesn't have noise
    im1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    # Read Noisy image
    im2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    # Find image shape
    [M, N] = im1.shape
    # Calc Worst Case Absolute Error
    WCAE = np.max(np.max(np.abs(im2 - im1)))
    print(np.array(WCAE))
    return WCAE
# WCAE( './/EX4-DataSet//Original//1.jpg', './/EX4-DataSet//Noise_eliminated//remove_salt-pepper.jpg')   #---------->214
# WCAE( './/EX4-DataSet//Original//2.jpg', './/EX4-DataSet//Noise_eliminated//remove_gaussian.jpg')  #---------->211
# WCAE('.//EX4-DataSet//Original//3.jpg', './/EX4-DataSet//Noise_eliminated//remove_periodic_noise.jpg')  #---------->255
# WCAE('.//EX4-DataSet//Original//4.jpg', './/EX4-DataSet//Noise_eliminated//remove_speckle.jpg')  #---------->255

#Comparison function SSIM
def SSIM(path1 , path2):
    image_orginal = cv2.imread(path1, 0)
    image_noise = cv2.imread(path2, 0)
    # Convert to Double
    image_orginal = cv2.normalize(image_orginal.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    # Find Shape of image 1 (Noiseless image)
    # Convert to Double
    image_noise = cv2.normalize(image_noise.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    # Local win for statistics
    win = np.ones(8)
    # Constant in the SSIM index formula
    C1 = 25
    C2 = 25
    C3 = 25
    win = win / np.sum(win[:])
    # Average of image 1
    M1 = cv2.filter2D(image_orginal, -1, win, borderType=cv2.BORDER_REPLICATE)
    M1p2 = np.multiply(M1, M1)
    # Average of image 2
    M2 = cv2.filter2D(image_noise, -1, win, borderType=cv2.BORDER_REPLICATE)
    M2p2 = np.multiply(M2, M2)
    M1_D_M2 = np.multiply(M1, M2)
    # Variance of image 1
    Var1p2 = cv2.filter2D(np.multiply(image_orginal, image_orginal), -1, win, borderType=cv2.BORDER_REPLICATE)
    # Variance of image 2
    Var2p2 = cv2.filter2D(np.multiply(image_noise, image_noise), -1, win, borderType=cv2.BORDER_REPLICATE)
    # Covariance of image 2
    Var12p2 = cv2.filter2D(np.multiply(image_orginal, image_noise), -1, win, borderType=cv2.BORDER_REPLICATE)
    SSIMMap = ((2 * M1_D_M2 + C1) / (M1p2 + M2p2 + C1)) * \
              ((2 * np.sqrt(Var1p2) * np.sqrt(Var2p2) + C2) / (Var1p2 + Var2p2 + C2)) * \
              ((Var12p2 + C3) / (np.sqrt(Var1p2) * np.sqrt(Var2p2) + C3))
    msssim = np.mean(np.mean(SSIMMap))
    print(msssim)
    return msssim
#
# SSIM( './/EX4-DataSet//Original//1.jpg', './/EX4-DataSet//Noise_eliminated//remove_salt-pepper.jpg')   #---------->
# SSIM( './/EX4-DataSet//Original//2.jpg', './/EX4-DataSet//Noise_eliminated//remove_gaussian.jpg')  #---------->
# SSIM('.//EX4-DataSet//Original//3.jpg', './/EX4-DataSet//Noise_eliminated//remove_periodic_noise.jpg')  #---------->
# SSIM('.//EX4-DataSet//Original//4.jpg', './/EX4-DataSet//Noise_eliminated//remove_speckle.jpg')  #---------->
#
#
#
def FOM(path1, path2):
    # Read Original image --> this image doesn't have noise
    im1 = cv2.normalize(cv2.imread(path1, cv2.IMREAD_GRAYSCALE).astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    # Find Shape of image 1 (Noiseless image)
    [m, n] = im1.shape
    # Read Noisy Image
    im2 = cv2.normalize(cv2.imread(path2, cv2.IMREAD_GRAYSCALE).astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    # Resize image 2 to size image 1
    im2 = cv2.resize(im2, [m, n])
    epsilon = np.finfo(np.float32).eps
    # Edge detector on images
    RefIm = canny(im1, 0.1, 20, 50)
    TestIm = canny(im2, 0.1, 20, 50)
    # Number of edge pixels in the test image
    C = np.sum(TestIm[:])
    # Number of edge pixels in the originl image
    B = np.sum(RefIm[:])
    Landa = 1.0 / 9
    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(np.invert(RefIm))
    fom = 1.0 / (epsilon + np.maximum(
        np.count_nonzero(RefIm),
        np.count_nonzero(TestIm)))
    N, M = im1.shape
    for i in range(N):
        for j in range(M):
            if RefIm[i, j]:
                fom += 1.0 / (1.0 + dist[i, j] * dist[i, j] * Landa)
    fom /= (epsilon + np.maximum(
        np.count_nonzero(RefIm),
        np.count_nonzero(TestIm)))
    print(fom)
    return fom


FOM( './/EX4-DataSet//Original//1.jpg', './/EX4-DataSet//Noise_eliminated//remove_salt-pepper.jpg')   #---------->0
FOM( './/EX4-DataSet//Original//2.jpg', './/EX4-DataSet//Noise_eliminated//remove_gaussian.jpg')  #---------->0
FOM('.//EX4-DataSet//Original//3.jpg', './/EX4-DataSet//Noise_eliminated//remove_periodic_noise.jpg')  #---------->43272
FOM('.//EX4-DataSet//Original//4.jpg', './/EX4-DataSet//Noise_eliminated//remove_speckle.jpg')  #---------->
