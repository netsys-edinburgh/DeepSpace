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
import os
import json
import matplotlib.pyplot as plt
import pdb

def load_meta_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def load_coordinates(path):
    meta_data = load_meta_json(path)
    # Load the coordinates, under the fieled: -> "geometry" -> "coordinates"
    return np.array(meta_data["geometry"]["coordinates"])

def load_clouds(path, thr, thr2):
    meta_data = load_meta_json(path)
    cloud_score = meta_data["properties"]["cloud_cover"]
    cloud_persent = meta_data["properties"]["cloud_percent"]
    if cloud_score > thr and cloud_persent > thr2:
        print(f'Threshold {thr}, File name:', os.path.split(path)[-1], 'Cloud score:', cloud_score, 'Cloud percent:', cloud_persent)
        return True
    else:
        return False

def load_trace(path):
    # get all the files in the directory that are .json
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    # sort the files by date. The date is in the file name YYYYMMDD_HHMMSS_rest_filename
    files.sort()
    # Load all the coordinates
    list_of_coordinates = [load_coordinates(os.path.join(path, f)) for f in files]
    return list_of_coordinates

def load_clouds_series(path, thr, thr2):
    # get all the files in the directory that are .tif
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    # sort the files by date. The date is in the file name YYYYMMDD_HHMMSS_rest_filename
    files.sort()
    # Load all the clouds
    list_of_clouds = [load_clouds(os.path.join(path, f), thr, thr2) for f in files]
    return list_of_clouds

def generate_color(num, mode='rainbow'):
    if mode == 'random':
        return np.random.rand(num, 3)
    elif mode == 'rainbow':
        return plt.cm.rainbow(np.linspace(0, 1, num))
    elif mode == 'jet':
        return plt.cm.jet(np.linspace(0, 1, num))
    elif mode == 'hsv':
        return plt.cm.hsv(np.linspace(0, 1, num))
    elif mode == 'nipy_spectral':
        return plt.cm.nipy_spectral(np.linspace(0, 1, num))
    elif mode == 'winter':
        return plt.cm.winter(np.linspace(0, 1, num))
    else:
        raise ValueError('Mode not recognized')
    
def bound4points(points):
    # find the bound of the points
    # points: np.array, shape (4, 2)
    # return: np.arrays, shape -> x-axis: (4,), x-axis-1: (4,), y-value-1: (4,)
    y_max = np.max(points[:,1])
    y_min = np.min(points[:,1])
    y_max_x = points[np.argmax(points[:,1]), 0]
    y_min_x = points[np.argmin(points[:,1]), 0]
    # x0-y0 is the lower curve, x1-y1 is the upper curve
    # x0 include all sorted x values except y_max_x
    # y0 includes all y sorted by according to x0 values except y_max
    # x1 include all sorted x values except y_min_x
    # y1 includes all y sorted by according to x1 values except y_min
    x0 = np.sort(points[points[:,1] != y_max, 0])
    x1 = np.sort(points[points[:,1] != y_min, 0])
    y0 = np.array([points[points[:,0] == x, 1] for x in x0]).flatten()
    y1 = np.array([points[points[:,0] == x, 1] for x in x1]).flatten()
    #merge and sort x0 and x1
    x = np.sort(np.concatenate([x0, x1]))
    # Nearest interpolation y0 and y1
    y0 = np.interp(x, x0, y0)
    y1 = np.interp(x, x1, y1)
    return x, y0, y1
    

def plot_trace(trace, separate_neighbours=False, mode='rainbow', save_path=None):
    # Plot the trace
    # color the neighbours differently
    plt.figure(figsize=(10, 10))
    colors = generate_color(len(trace), mode=mode)
    for i, coordinates in enumerate(trace):
        if separate_neighbours:
            for j in range(5):
                plt.scatter(coordinates[0,j,0], coordinates[0,j,1], c=colors[i])
        else:
            for j in range(5):
                plt.scatter(coordinates[0,j,0], coordinates[0,j,1])
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_coverage(trace, mode='rainbow', save_path=None):
    # Plot the trace
    # color the neighbours differently
    plt.figure(figsize=(10, 10))
    colors = generate_color(len(trace), mode=mode)
    for i, coordinates in enumerate(trace):
        x, y0, y1 = bound4points(coordinates[0,:4,:])
        plt.fill_between(x, y0, y1, color=colors[i], alpha=0.1)
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def if_overlap(co0, co1, num=100000):
    # check if the two coordinates overlap
    # return ratio of overlap in the first coordinate
    # co0, co1: np.array, shape (4, 2), the coordinates of the two polygons
    # return: float, float, the ratio of overlap in the first coordinate and the ratio of overlap in the second coordinate
    inpolygon = lambda x, y, co: np.sum((co[:,0] - x) * (co[:,1] - y) > 0) % 2

    # Monte Carlo simulation, estimate the overlap
    # num = 100000 --- number of points

    # generate random points
    x = np.random.uniform(np.min(co0[:,0]), np.max(co0[:,0]), num)
    y = np.random.uniform(np.min(co0[:,1]), np.max(co0[:,1]), num)

    # check if the points are in the polygon
    in0 = np.array([inpolygon(x[i], y[i], co0) for i in range(num)])
    in1 = np.array([inpolygon(x[i], y[i], co1) and inpolygon(x[i], y[i], co0) for i in range(num)])
    if np.sum(in0) == 0 or np.sum(in1) == 0:
        print('No overlap in the first coordinate, return 0, 0')
        return 0, 0
    else:
        overlap_0 = min(np.sum(in1) / np.sum(in0), 1)

    # generate random points
    x = np.random.uniform(np.min(co1[:,0]), np.max(co1[:,0]), num)
    y = np.random.uniform(np.min(co1[:,1]), np.max(co1[:,1]), num)

    # check if the points are in the polygon
    in0 = np.array([inpolygon(x[i], y[i], co0) and inpolygon(x[i], y[i], co1) for i in range(num)])
    in1 = np.array([inpolygon(x[i], y[i], co1) for i in range(num)])
    if np.sum(in0) == 0 or np.sum(in1) == 0:
        print('No overlap in the second coordinate, return 0, 0')
        return 0, 0
    else:
        overlap_1 = min(np.sum(in0) / np.sum(in1),1)
    return overlap_0, overlap_1

def px2coord(px, img, co):
    # convert the pixel to coordinate
    # px: np.array, shape (2,), the pixel
    # img: np.array, shape (H, W), the image
    # co: np.array, shape (4, 2), the coordinates
    # return: np.array, shape (2,), the coordinate
    H, W = img.shape
    x = co[:,0]
    y = co[:,1]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    x = px[0] / W * (x_max - x_min) + x_min
    y = px[1] / H * (y_max - y_min) + y_min
    return np.array([x, y])

def overlap_mask(pxs, coordinates):
    # generate the overlap mask
    # pxs: np.array, shape (N, 2), the pixels
    # coordinates: np.array, shape (2, 4, 2), the coordinates
    # return: np.array, shape (N,), the mask
    
    # check if the two coordinates overlap in pixel space
    inpolygon = lambda x, y, co: np.sum((co[:,0] - x) * (co[:,1] - y) > 0) % 2
    inoverlap = lambda x, y, co0, co1: inpolygon(x, y, co0) and inpolygon(x, y, co1)

    mask = np.zeros(pxs.shape[0])
    for i, px in enumerate(pxs):
        mask[i] = inpolygon(px[0], px[1], coordinates[0]) and inpolygon(px[0], px[1], coordinates[1])
    return mask
    
    
def export_overlap(imgs, coordinates, save_path):
    # export the overlap of the two images

    # imgs: list of np.array, shape (H, W), the two images
    img0 = imgs[0]
    img1 = imgs[1]
    co0 = coordinates[0]
    co1 = coordinates[1]

    # check if the two coordinates overlap
    overlap_0, overlap_1 = if_overlap(co0, co1)
    size0 = img0.shape[0] * img0.shape[1]
    size1 = img1.shape[0] * img1.shape[1]

    # calculate the overlap
    overlap_0 = overlap_0 * size0 
    overlap_1 = overlap_1 * size1

    # export the overlap
    if overlap_0 > 160000 and overlap_1 > 160000:
        # process the overlap

        # find the overlap region in the first image
        # convert the coordinates to pixels: 
        # (1) first convert pixels to coordinates.
        # (2) find the overlap region mask.
        # (3) extract the overlap image.
        copx0 = px2coord(np.array([0, 0]), img0, co0)
        copx1 = px2coord(np.array([0, 0]), img1, co1)
    return