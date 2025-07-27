import numpy as np
from PIL import Image
import os
# import cv2

'''
This file contains functions to extract information from images
We elaborate on the functions in the file to make them more useful
All the analysis will be jointly used to detect near-duplicates, image anomalies, and other image-related tasks
'''

def compute_entropy(path):
    """
    Compute the entropy of an image
    """
    img = Image.open(path)
    histogram = img.histogram()
    histogram_length = sum(histogram)
    samples_probability = [float(h) / histogram_length for h in histogram]
    return -sum([p * np.log2(p) for p in samples_probability if p != 0])

def color_std(path):
    """
    Compute the standard deviation of the color values
    """
    img = Image.open(path)
    img = np.array(img)
    return [img[:, :, i].std() for i in range(3)]


def filesize(path):
    """
    Return the size of a file in bytes
    """
    return os.path.getsize(path)

# def edge_detection_sobel(path):
#     """
#     Compute the edge detection using the Sobel operator
#     Return number of edges
#     """
#     img = Image.open(path).convert('L')
#     img = np.array(img)
#     img = np.float32(img)
#     gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
#     gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
#     mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
#     return np.sum(mag > 100)

def color_analysis(path):
    """
    Compute the color analysis of an image
    Return the average color value
    """
    img = Image.open(path)
    img = img.resize((64, 64))
    img = np.array(img)
    return [img[:, :, i].mean() for i in range(3)]

def spectral_analysis(path):
    '''
    1. do Fourier transform
    2. shift the zero frequency component to the center
    3. compute the magnitude of the spectrum
    4. normalize the magnitude to the range [0, 1]

    '''
    img = np.array(img)
    spectrum = np.fft.fft2(img, axes=(0, 1))
    # Shift the zero frequency component to the center
    spectrum = np.fft.fftshift(spectrum, axes=(0, 1))
    # Compute the magnitude of the spectrum
    magnitude = np.abs(spectrum)
    # Normalize the magnitude to the range [0, 1]
    magnitude -= magnitude.min()
    magnitude /= magnitude.max()
    hist_list = []
    for i in range(spectrum.shape[2]-1):
        hist, bins = np.histogram(np.abs(spectrum[:,:,i]).ravel(), bins=20, range=(20, 100000))
        hist_list.append(hist)
    return hist_list

def high_freq_analysis(path):
    '''
    1. do Fourier transform
    2. shift the zero frequency component to the center
    3. compute the magnitude of the spectrum
    4. normalize the magnitude to the range [0, 1]

    '''
    img = Image.open(path)
    img = np.array(img)
    spectrum = np.fft.fft2(img, axes=(0, 1))
    # Shift the zero frequency component to the center
    spectrum = np.fft.fftshift(spectrum, axes=(0, 1))
    # Compute the magnitude of the spectrum
    magnitude = np.abs(spectrum)
    # Normalize the magnitude to the range [0, 1]
    magnitude -= magnitude.min()
    magnitude /= magnitude.max()
    hist_list = []
    for i in range(spectrum.shape[2]-1):
        hist, bins = np.histogram(np.abs(spectrum[:,:,i]).ravel(), bins=20, range=(20, 100000))
        hist_list.append(np.sum(hist[10:]))
    return np.sum(hist_list)

def freq_std(path):
    '''
    1. do Fourier transform
    2. shift the zero frequency component to the center
    3. compute the magnitude of the spectrum
    4. normalize the magnitude to the range [0, 1]

    '''
    img = Image.open(path)
    img = np.array(img)
    spectrum = np.fft.fft2(img, axes=(0, 1))
    # Shift the zero frequency component to the center
    spectrum = np.fft.fftshift(spectrum, axes=(0, 1))
    # Compute the magnitude of the spectrum
    magnitude = np.abs(spectrum)
    # Normalize the magnitude to the range [0, 1]
    magnitude -= magnitude.min()
    magnitude /= magnitude.max()
    return [np.std(magnitude[:,:,i]) for i in range(3)]
