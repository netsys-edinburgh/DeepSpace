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
import numpy as np
#import imutils # pip install imutils
from PIL import Image # pip install pillow
import tifffile
from matplotlib import pyplot as plt
import os
import sys
import pdb
from skimage.metrics import structural_similarity as ssim

def rotate_image(img):
    #scale to 0-255
    scale_factor=255/np.max(img)
    img = img * scale_factor

    #let's remove all the initial rows and columns that are all zeros
    rows = 0
    while np.all(img[rows] == 0):
        rows += 1
    cols = 0
    while np.all(img[:,cols] == 0):
        cols += 1

    img = img[rows:,cols:]
    #now let's rotate the image so that the first non-zero pixel is in the top left
    firstX = np.nonzero(img[0, :, :])[0][0]
    firstY = np.nonzero(img[:, 0, :])[0][0]
    angle = -np.arctan2(firstX, firstY)

    #img = imutils.rotate_bound(img, angle*180/np.pi)
    # convert to JPG image from tiff image
    
    img = Image.fromarray(img.astype(np.uint8)) # convert to PIL image
    img = img.rotate(-angle*180/np.pi, expand=True)
    new_image = np.array(img)

    #let's crop all the rows and columns that are all zeros
    rows = 0
    while np.all(new_image[rows] == 0):
        rows += 1
    cols = 0
    while np.all(new_image[:,cols] == 0):
        cols += 1
    
    new_image = new_image[rows:,cols:]
    #now let's remove the rows and columns that are all zeros from the other side
    rows = new_image.shape[0] - 1
    while np.all(new_image[rows] == 0):
        rows -= 1
    cols = new_image.shape[1] - 1
    while np.all(new_image[:,cols] == 0):
        cols -= 1
    new_image = new_image[:rows+1,:cols+1]

    #now let's rotate the image if x is greater than y
    if new_image.shape[0] > new_image.shape[1]:
        new_image = np.rot90(new_image)
    
    print(new_image.shape)

    return new_image/scale_factor

def creat_batch(image, batch_size, step_size=0):
    # cut the input image into batch_size parts
    # automatically skip the batch with any zero(s) in it
    # If step_size is not zero, then the batch move step_size pixels each time, horizontally then vertically
    # the small batch is [batch_size, batch_size, n_channels]
    # will not append zero padding to the image but just skip the batch when the pixels are not enough
    # image: numpy array
    # batch_size: int
    # return: list of numpy arrays
    # step_size: int

    if step_size == 0:
        step_size = batch_size
    else:
        step_size = step_size
    batches = []
    for i in range(0, image.shape[0]-batch_size+1, step_size):
        for j in range(0, image.shape[1]-batch_size+1, step_size):
            batch = image[i:i+batch_size, j:j+batch_size, :]
            #if np.min(batch[:,:,0]+batch[:,:,1]+batch[:,:,2])>0:
            # >95% of the pixels are not zero
            if np.sum(batch[:,:,0]+batch[:,:,1]+batch[:,:,2]>0) > 0.95*batch_size*batch_size:
                batches.append(batch)
    return batches

def loadandplot(path, title=''):
  a=tifffile.imread(path)
  plt.figure(figsize=(5,5))
  print(a.shape)
  plt.imshow(a[:,:,[2,1,0]]/np.max(a[:,:,:3]))
  if title:
     plt.title(title)

def loadandsave(path, batchsize, save_path):
  a = tifffile.imread(path)
  a = a[:,:,[2,1,0]]/np.max(a[:,:,:3])
  batches = creat_batch(a, batchsize)
  np.save(save_path, batches)

def loadandsaveaspng(path, batchsize, save_path):
    tiff = tifffile.imread(path)
    tiff = tiff[:,:,[2,1,0]]
    batches = creat_batch(tiff, batchsize)

    for i in range(len(batches)):
        # im = Image.fromarray(np.array(batches[i]).astype(np.uint8))
        # im.save(save_path + '_' + str(i) + '.png')
        # using matplotlib
        plt.imshow(batches[i]/np.max(batches[i]))
        plt.axis('off')
        plt.savefig(save_path + '_' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()

def ssim_tolerance(img, width, tolerance):
    # resize img to widthxwidth, and calculate the SSIM between the resized image and the original image
    # if SSIM > tolerance, return True, else return False
    # img: np.array
    # width: int
    # tolerance: float
    # return: bool
    normalized_image  = (img - np.min(img))/(np.max(img) - np.min(img))
    img = (normalized_image * 255).astype(np.uint8)
    img = Image.fromarray(img)
    # downsample the image
    img2 = img.resize((width, width))
    # resize the image back to the original size, this is the lower bound for the SSIM
    img2 = img2.resize(img.size)
    img = np.array(img)
    img2 = np.array(img2)
    img = img/np.max(img)
    img2 = img2/np.max(img2)
    return ssim(img, img2, multichannel=True) > tolerance

def loadandsavepng_ssim(path, batchsize, basecr, save_path, file_name, threshold=0.9):
    '''
    path: str, the path of the tiff file
    batchsize: int, the size of the batch
    save_path: str, the path to save the png files
    basecr: float, the base compression rate in the resize
    '''
    # resizze the image to 1/basecr, and resize back to the original size
    # calculate the SSIM between the original image and the resized image
    # save original image to correspoinding path matches SSIM value: SSIM > 0.9 to save_path + 'green_basecr/', SSIM < 0.9 to save_path + 'red_basecr/'
    
    # create the save_path + 'green_basecr/' and save_path + 'red_basecr/' if not exist
    if not os.path.exists(save_path + 'deepgreen/'):
        os.makedirs(save_path + 'deepgreen/')
    if not os.path.exists(save_path + 'deepred/'):
        os.makedirs(save_path + 'deepred/')

    tiff = tifffile.imread(path)
    tiff = tiff[:,:,[2,1,0]]
    batches = creat_batch(tiff, batchsize)

    for i in range(len(batches)):
        # check the SSIM tolerance of batches[i]
        if ssim_tolerance(batches[i], int(batchsize/basecr**0.5), threshold):
            # save the image to save_path + 'green_basecr/'
            plt.imshow(batches[i]/np.max(batches[i]))
            plt.axis('off')
            plt.savefig(save_path + 'deepgreen/' + file_name + '_' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            #pass
        else:
            # save the image to save_path + 'red_basecr/'
            plt.imshow(batches[i]/np.max(batches[i]))
            plt.axis('off')
            plt.savefig(save_path + 'deepred/' + file_name + '_' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        

def loadandplotgrey(path):
  a=tifffile.imread(path)
  plt.figure(figsize=(5,5))
  plt.imshow(a[:,:,[2,1,0]]/np.max(a[:,:,:3]), cmap='gray')

# load the files end with _Analytic.tif, and size larger than 1000x1000
def loadfiles(path):
  files = os.listdir(path)
  files = [f for f in files if f.endswith('_Analytic.tif')]
  files = [f for f in files if tifffile.imread (path + f).shape[0] > 1000]  
  return files

def loadfiles_dy(path):
  files = os.listdir(path)
  files = [f for f in files if f.endswith('.tif')]
  files = [f for f in files if tifffile.imread (path + f).shape[0] > 1000]  
  return files

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