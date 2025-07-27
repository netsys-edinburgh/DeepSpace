from mycompress import *
from matplotlib import pyplot as plt
from skimage import io, color
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import os
import sys
import pdb

def compute_ssim(img1, img2):
    # Ensure the images have the same shape
    print(img1.shape)
    print(img2.shape)
    assert img1.shape == img2.shape, "The images must have the same size and number of channels"

    # Compute SSIM for each channel
    ssim_r = compare_ssim(img1[:,:,0], img2[:,:,0])
    ssim_g = compare_ssim(img1[:,:,1], img2[:,:,1])
    ssim_b = compare_ssim(img1[:,:,2], img2[:,:,2])

    # Average the SSIM values of the channels
    ssim_avg = (ssim_r + ssim_g + ssim_b) / 3

    return ssim_avg

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def loadfiles(path):
    # load the files into a list
    files = []
    for file in os.listdir(path):
        if file.endswith('.npy') and 'processed' not in file:
            files.append(file)
    return files
# path = '/mnt/data/chuanhao/segment-anything/amazon/planet/planet/train-jpg/train_44.jpg'

class IMGCOMPRESS:  
     
     def __init__(self, method, cr, filter_method = 'LANCZOS', sensing_method = 'quantile'):
        self.method = method
        self.cr = cr
    
        # check the method
        assert self.method in ['resize', 'sensing'], "The method must be either 'resize' or 'sensing'"

        # check and define the filter
        # By default, the filter is set to 'LANCZOS', which is the highest quality filter
        # the Nearest filter is the fastest but the lowest quality
        # Note that the filter is only used when the method is 'resize'
        if self.method == 'resize':
            assert filter_method in ['NEAREST', 'BOX', 'BILINEAR', 'HAMMING', 'BICUBIC', 'LANCZOS'], "The filter must be one of the following: 'NEAREST', 'BOX', 'BILINEAR', 'HAMMING', 'BICUBIC', 'LANCZOS'"
            if filter_method == 'NEAREST':
                filter = Image.NEAREST
            elif filter_method == 'BOX':
                filter = Image.BOX
            elif filter_method == 'BILINEAR':
                filter = Image.BILINEAR
            elif filter_method == 'HAMMING':
                filter = Image.HAMMING
            elif filter_method == 'BICUBIC':
                filter = Image.BICUBIC
            elif filter_method == 'LANCZOS':
                filter = Image.LANCZOS
            self.filter = filter
            self.filter_method = filter_method

        elif self.method == 'sensing':
            assert sensing_method in ['quantile', 'random'], "The sensing method must be either 'quantile' or 'random'"
            self.sensing_method = sensing_method
    
     def compress(self, img):
        # Scale the image to 0-255
        img[-1, -1, 0] = 0
        img[-1, -1, 1] = 0
        img[-1, -1, 2] = 0
        img[-2, -2, 0] = 255
        img[-2, -2, 1] = 255
        img[-2, -2, 2] = 255

        # Read the image and separate RGB channels
        ch0 = img[:,:,0]
        ch1 = img[:,:,1]
        ch2 = img[:,:,2]

        # Generate a random sampling mask
        if self.sensing_method == 'random' and self.method == 'sensing':
            mask = random_sampling_mask(ch0.shape, 1-1/self.cr)
            mode = 'keep'
            self.cr = 1000
        else:
            mask = random_sampling_mask(ch0.shape, 0.999)
            mode = 'remove'
       
        # Compress the image
        
        # Perform the compression based on the method
        if self.method == 'resize':
            # Resize the image
            compressed_image = Image.fromarray(img).resize((int(img.shape[1]/self.cr), int(img.shape[0]/self.cr)), resample=self.filter)

            #Resize back to original size
            compressed_image = compressed_image.resize((img.shape[1], img.shape[0]), resample=self.filter)

            # Convert the resized image to a numpy array
            compressed_image = np.array(compressed_image)
        
        # Perform frequency domain compression
        elif self.method == 'sensing':
             compressed_ch0 = one_bit_quantization(ch0, mask, self.cr, mode)
             compressed_ch1 = one_bit_quantization(ch1, mask, self.cr, mode)
             compressed_ch2 = one_bit_quantization(ch2, mask, self.cr, mode)
             compressed_image = np.stack([np.abs(ifft2(compressed_ch0)), np.abs(ifft2(compressed_ch1)), np.abs(ifft2(compressed_ch2))], axis=-1).astype(np.uint8)

        return compressed_image
     
def process_data(files, method, cr, filter_method = 'LANCZOS', sensing_method = 'quantile'):
    # Create an instance of the IMGCOMPRESS class
    imgcompress = IMGCOMPRESS(method, cr, filter_method, sensing_method)

    # Create a list to store the SSIM values
    compressed_files = []

    # Loop through the files
    for file in files:
        # Load the image
        img = (file*255).astype(np.uint8)

        # Compress the image
        compressed_img = imgcompress.compress(img)

        # put the compressed image into the list
        compressed_files.append(compressed_img)

    return compressed_files


# run the code
def main():
    # Config
    cr = 100
    method = 'sensing'

    path = '/mnt/raid0sata1/chuanhao/CA/cropped'
    files = loadfiles(path)

    for npy_file in progressbar(files):
        npy_list = np.load(os.path.join(path, npy_file), allow_pickle=True)
        compressed_files = process_data(npy_list, method, cr, 'LANCZOS', 'quantile')
        np.save(os.path.join(path, npy_file.replace('.npy', f'_processed_{cr}_{method}.npy')), compressed_files)

if __name__ == '__main__':
    main()


