'''
/*
    * This file is part of DeepSpace.
    *
    * DeepSpace is free software: you can redistribute it and/or modify
    * it under the terms of the GNU Affero General Public License as published by
    * the Free Software Foundation, either version 3 of the License, or
    * (at your option) any later version.
    *
    * DeepSpace is distributed in the hope that it will be useful,
    * but WITHOUT ANY WARRANTY; without even the implied warranty of
    * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    * GNU Affero General Public License for more details.
    *
    * You should have received a copy of the GNU Affero General Public License
    * along with DeepSpace.  If not, see <https://www.gnu.org/licenses/>.
    */
'''
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import cvxpy as cp
from skimage import io, color
from numpy.fft import fft2, ifft2
import scipy.fftpack as spfft
import matplotlib.image as mpimg


dpath = '/mnt/data/chuanhao/segment-anything/amazon/planet/planet/train-jpg/train_84.jpg'
# Load the image
# def load_image_and_masks(path=dpath):
#     #path = 'shipsnet/shipsnet/0__20161218_180845_0e26__-122.32608603844608_37.72807687984217.png'
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Load the image
#     # image = Image.open(path)
#     # image = np.array(image)

#     sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
#     mask_generator = SamAutomaticMaskGenerator(sam)
#     masks = mask_generator.generate(image)
#     return image, masks

def mask_shape(mask):
    #Get hight of mask
    for i in range(mask.shape[0]):
        if np.sum(mask[i, :]) > 0:
            h_0 = i
            break
    for i in range(h_0, mask.shape[0]):
        if np.sum(mask[i, :]) == 0:
            h_1 = i
            break
        h_1 = mask.shape[0]
    #Get width of mask
    for i in range(mask.shape[1]):
        if np.sum(mask[:, i]) > 0:
            w_0 = i
            break
    for i in range(w_0, mask.shape[1]):
        if np.sum(mask[:, i]) == 0:
            w_1 = i
            break
        w_1 = mask.shape[1]
    #get size of mask
    ms = np.sum(mask, axis=(0, 1))
    return h_0, h_1, w_0, w_1, ms

def show_anns(anns, show_background=False):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        #pdb.set_trace()
        if np.sum(np.sum(m)) < 60000:
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
            h_0, h_1, w_0, w_1, ms = mask_shape(m)
            ax.text((w_0+w_1)/2, (h_0+h_1)/2, f'Region{idx}::'+str(ms), fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
        elif show_background:
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
    ax.imshow(img)

def modulation(img, anns, region):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    if isinstance(region, int):
        m = sorted_anns[region]['segmentation']
        random_array = np.random.uniform(0, 255, (m.shape[0], m.shape[1]))
        for idx in range(3):
            img[m,idx] = random_array[m]
    else:
        assert isinstance(region, list), 'region must be a list'
        for idx in region:
            m = sorted_anns[idx]['segmentation']
            random_array = np.random.uniform(0, 255, (m.shape[0], m.shape[1]))
            for channel in range(3):
                img[m,channel] = random_array[m]
    return img

def random_sampling_mask(shape, fraction):
    """Generate a random sampling mask for the given shape."""
    total_elements = shape[0] * shape[1]
    sample_count = int(total_elements * fraction)
    mask = np.zeros(total_elements, dtype=np.uint8)
    mask[:sample_count] = 1
    np.random.shuffle(mask)
    return mask.reshape(shape)

def one_bit_quantization(channel, mask, cr=10, mode='remove'):
    """Perform one-bit quantization on the channel."""
    # Transform channel to spectral domain
    spectral_domain = fft2(channel)
    
    # Support mask - remove: remove the components that are not sampled
    assert mode in ['remove', 'keep'], 'mode must be either remove or keep'

    if mode == 'remove':
        # Randomly sample in the spectral domain
        sampled = spectral_domain * mask
    elif mode == 'keep':
        to_add_back =  spectral_domain * mask
        sampled = spectral_domain * (1 - mask)

    # Further downsample the sampled spectral domain representation by quantail
    # get quantail
    assert cr > 1, 'compression ratio must be greater than 1'
    quantail = np.quantile(np.abs(sampled), 1 - 1/cr)
    sampled[np.abs(sampled) < quantail] = 0

    if mode == 'keep':
        sampled += to_add_back

    # print actual CR
    # print(f'Actual CR: {sampled.shape[0] * sampled.shape[1] / np.sum(np.abs(sampled) > 0)}')

    # check if CR is met
    # CR = total_size/none_zero_num
    # assert sampled.shape[0] * sampled.shape[1] / np.sum(np.abs(sampled) > 0) >= cr*0.9, 'compression ratio not met'

    return sampled

def compressive_sensing_reconstruction(channel, mask, mode='direct', cr=10, mask_mode='remove'):
    """Reconstruct an image channel using compressive sensing."""
    # Transform channel to spectral domain
    # Randomly sample in the spectral domain

    assert mode in ['direct', 'quantile'], 'mode must be either direct or quantile'
    if mode == 'direct':
        spectral_domain = fft2(channel)
        sampled = spectral_domain * mask
    elif mode == 'quantile':
        sampled = one_bit_quantization(channel, mask, cr, mask_mode)
    # Vectorize
    y = sampled.flatten()
    
    # Create the measurement matrix
    # Phi = (np.eye(len(y))!=0).T
    # Obvervation (sensing) method is droping values, so Phi is a diagonal matrix
    if mask_mode == 'remove':
        Phi = np.diag(mask.flatten())
    else:
        Phi = np.diag((1-mask).flatten())
    
    # Set up the L1 optimization problem
    x = cp.Variable(len(y), complex=True)
    constraints = [Phi @ x == y]
    objective = cp.Minimize(cp.norm(x, 1))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # Get the reconstructed spectral domain representation and transform back to spatial
    reconstructed_spectral = np.array(x.value).reshape(channel.shape)
    return np.abs(ifft2(reconstructed_spectral))

def plot_spectrum(channel, ax, title):
    """
    Compute the 2D Fourier Transform of the channel and plot its magnitude spectrum.
    """
    # Compute 2D FFT
    f = np.fft.fft2(channel)
    
    # Shift zero frequency component to the center
    fshift = np.fft.fftshift(f)
    
    # Compute magnitude spectrum
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    
    ax.imshow(magnitude_spectrum, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

def plot_spectrum_directly(channel, ax, title):
    #turn into amplitude 
    
    #shift zero frequency component to the center
    fshift = np.fft.fftshift(channel)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    ax.imshow(magnitude_spectrum, cmap='gray')
    ax.set_title(title)
    ax.axis('off')


def plot_comparison_spectrum(channel, mask, cr=10, mode='remove'):
    """
    Plot the magnitude spectrum of the original channel and the reconstructed channel.
    """
    # Create a subplot to show the spectrum for each channel
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the original channel spectrum
    plot_spectrum(channel, axs[0], 'Original Spectrum')
    
    # Perform compressive sensing reconstruction
    sampled_channel = one_bit_quantization(channel, mask, cr, mode)
    
    # Plot the reconstructed channel spectrum
    plot_spectrum_directly(sampled_channel, axs[1], 'Sampled Spectrum')
    
    plt.tight_layout()
    plt.show()
    return sampled_channel


def img_spectra(path=dpath):
    # Load the RGB image
    img = mpimg.imread(path)
    
    # Extract R, G, B channels
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    # Create a subplot to show the spectrum for each channel
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    plot_spectrum(R, axs[0], 'Red Channel Spectrum')
    plot_spectrum(G, axs[1], 'Green Channel Spectrum')
    plot_spectrum(B, axs[2], 'Blue Channel Spectrum')
    
    plt.tight_layout()
    plt.show()