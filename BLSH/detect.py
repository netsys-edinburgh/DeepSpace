import argparse
import sys
import os
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Tuple

import imagehash
import numpy as np
from PIL import Image

import pdb


def calculate_signature(image_file: str, hash_size: int) -> np.ndarray:
    """ 
    Calculate the dhash signature of a given file
    
    Args:
        image_file: the image (path as string) to calculate the signature for
        hash_size: hash size to use, signatures will be of length hash_size^2
    
    Returns:
        Image signature as Numpy n-dimensional array or None if the file is not a PIL recognized image
    """
    pil_image = Image.open(image_file).convert("L").resize(
                        (hash_size+1, hash_size),
                        Image.Resampling.LANCZOS) # Resize to (hash_size+1, hash_size) to get hash_size^2 bits, Image.ANTIALIAS is deprecated
    dhash = imagehash.dhash(pil_image, hash_size)
    '''
    Resize method:
    PIL.Image.NEAREST by default
    PIL.Image.BOX for box resampling
    PIL.Image.BILINEAR for bilinear resampling
    PIL.Image.HAMMING for hamming resampling
    PIL.Image.BICUBIC for bicubic resampling
    PIL.Image.LANCZOS for lanczos resampling
    '''
    signature = dhash.hash.flatten()
    pil_image.close()
    return signature

def hamming_distance(signature_0: np.ndarray, signature_1: np.ndarray) -> int:
    """
    Calculate the Hamming distance between two image signatures
    
    Args:
        signature_0: the first image signature
        signature_1: the second image signature
    
    Returns:
        The Hamming distance between the two signatures
    """
    return np.sum(np.bitwise_xor(signature_0, signature_1))

def similarity(signature_0: np.ndarray, signature_1: np.ndarray) -> float:
    """
    Calculate the similarity ratio between two image signatures
    
    Args:
        signature_0: the first image signature
        signature_1: the second image signature
    
    Returns:
        The similarity ratio between the two signatures
    """
    return 1 - hamming_distance(signature_0, signature_1) / len(signature_0)

def find_locality_strict_match(signature_0, signature_1, bands, rows):
    '''
    This function will be used to find the locality of the difference in 2D position
    The difference will be calculated by strict match
    If the difference is not zero, the position will be recorded
    '''
    locality_row = []
    for i in range(bands):
        signature_band_0 = signature_0[i*rows:(i+1)*rows]
        signature_band_1 = signature_1[i*rows:(i+1)*rows]
        if np.any(signature_band_0 != signature_band_1):
            locality_row.append(i)
    locality_col = []
    for i in range(rows):
        signature_band_0 = signature_0[i::rows]
        signature_band_1 = signature_1[i::rows]
        if np.any(signature_band_0 != signature_band_1):
            locality_col.append(i)
    return locality_row, locality_col

def find_locality_hdsim(signature_0, signature_1, bands, rows, threshold):
    '''
    This function will be used to find the locality of the difference in 2D position
    The difference will be calculated by Hamming distance and similarity
    If the difference is smaller than the threshold, the position will be recorded
    As a result of this function, we will have the locality of the difference
    '''
    locality_row = []
    for i in range(bands):
        signature_band_0 = signature_0[i*rows:(i+1)*rows]
        signature_band_1 = signature_1[i*rows:(i+1)*rows]
        if similarity(signature_band_0, signature_band_1) < threshold:
            locality_row.append(i)
    locality_col = []
    for i in range(rows):
        signature_band_0 = signature_0[i::rows]
        signature_band_1 = signature_1[i::rows]
        if similarity(signature_band_0, signature_band_1) < threshold:
            locality_col.append(i)
    return locality_row, locality_col

def locality2mask(locality_row, locality_col, bands, rows):
    '''
    This function will be used to convert the locality of the difference to a mask
    The mask will be used to highlight the difference in the image
    '''
    mask = np.zeros((bands*rows,), dtype=bool)
    for i in locality_row:
        mask[i*rows:(i+1)*rows] = True
    for i in locality_col:
        mask[i::rows] = True
    return mask

def highlight_difference(image_file_0: str, image_file_1: str, locality_row: List[int], locality_col: List[int], bands, rows) -> Tuple[Image.Image, Image.Image]:
    """
    Highlight the difference between two images
    
    Args:
        image_file_0: the first image file
        image_file_1: the second image file
        locality_row: the locality of the difference in the row
        locality_col: the locality of the difference in the column
        bands: the number of bands to use in the locality sensitve hashing process
        rows: the number of rows in the signature

    Returns:
        A tuple of the two images with the difference highlighted
    """
    img_0 = Image.open(image_file_0)
    img_1 = Image.open(image_file_1)
    img_0 = img_0.convert("RGBA")
    img_1 = img_1.convert("RGBA")
    img_0_data = img_0.getdata()
    img_1_data = img_1.getdata()
    new_img_0_data = []
    new_img_1_data = []
    # Recover the locality of the difference in actual image position with respect to the bands and rows
    locality_row = [i*bands for i in locality_row]
    locality_col = [i*rows for i in locality_col]
    for i, pixel in enumerate(img_0_data):
        if i//img_0.width in locality_row or i%img_0.width in locality_col:
            new_img_0_data.append((255, 0, 0, 255)) # highlight by red
        else:
            new_img_0_data.append(pixel)
    for i, pixel in enumerate(img_1_data):
        if i//img_1.width in locality_row or i%img_1.width in locality_col:
            new_img_1_data.append((255, 0, 0, 255))
        else:
            new_img_1_data.append(pixel)
    img_0.putdata(new_img_0_data)
    img_1.putdata(new_img_1_data)
    return img_0, img_1

        
def find_near_duplicates(input_dir: str, threshold: float, hash_size: int, bands: int) -> List[Tuple[str, str, float]]:
    """
    Find near-duplicate images
    
    Args:
        input_dir: Directory with images to check
        threshold: Images with a similarity ratio >= threshold will be considered near-duplicates
        hash_size: Hash size to use, signatures will be of length hash_size^2
        bands: The number of bands to use in the locality sensitve hashing process
        
    Returns:
        A list of near-duplicates found. Near duplicates are encoded as a triple: (filename_A, filename_B, similarity)
    """
    rows: int = int(hash_size**2/bands)
    signatures = dict()
    hash_buckets_list: List[Dict[str, List[str]]] = [dict() for _ in range(bands)]
    
    # Build a list of candidate files in given input_dir
    file_list = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f))]

    # remove the non-image files
    file_list = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

    # skip zeros files, the img should not be totally black (all zeros)
    file_list = [f for f in file_list if np.array(Image.open(f))[:,:,:-1].any()]

    print(f"Found {len(file_list)} images in {input_dir}")

    # Iterate through all files in input directory
    for fh in file_list:
        try:
            signature = calculate_signature(fh, hash_size)
        except IOError:
            # Not a PIL image, skip this file
            continue

        # Keep track of each image's signature
        # This is the part need to be sent to the satellite for each image
        signatures[fh] = np.packbits(signature) # convert to bytes, (256,) -> (32,)
        
        # Locality Sensitive Hashing (LSH) by rows
        for i in range(bands):
            signature_band = signature[i*rows:(i+1)*rows]
            signature_band_bytes = signature_band.tobytes()
            if signature_band_bytes not in hash_buckets_list[i]:
                hash_buckets_list[i][signature_band_bytes] = list()
            hash_buckets_list[i][signature_band_bytes].append(fh)
        

    # Build candidate pairs based on bucket membership
    candidate_pairs = set()
    for hash_buckets in hash_buckets_list:
        for hash_bucket in hash_buckets.values():
            if len(hash_bucket) > 1:
                hash_bucket = sorted(hash_bucket)
                for i in range(len(hash_bucket)):
                    for j in range(i+1, len(hash_bucket)):
                        candidate_pairs.add(
                            tuple([hash_bucket[i],hash_bucket[j]])
                        )

    # Check candidate pairs for similarity by Hamming distance
    near_duplicates = list()
    locality = dict()
    for cpa, cpb in candidate_pairs:
        hd = sum(np.bitwise_xor(
                np.unpackbits(signatures[cpa]), 
                np.unpackbits(signatures[cpb])
        ))
        similarity = (hash_size**2 - hd) / hash_size**2
        if similarity > threshold:
            near_duplicates.append((cpa, cpb, similarity))
    # save the location with the difference
    for a,b,s in near_duplicates:
        signature_0 = np.unpackbits(signatures[a])
        signature_1 = np.unpackbits(signatures[b])
        # threshold = 1-1/h_size^2, the maximum similarity for the locality
        locality[(a,b)] = find_locality_hdsim(signature_0, signature_1, bands, rows, 1-1/hash_size**2)
            
    # Sort near-duplicates by descending similarity and return
    near_duplicates.sort(key=lambda x:x[2], reverse=True)

    return near_duplicates, locality


def main(argv):
    # Argument parser
    parser = argparse.ArgumentParser(description="Efficient detection of near-duplicate images using locality sensitive hashing")
    parser.add_argument("-i", "--inputdir", type=str, default="", help="directory containing images to check")
    parser.add_argument("-t", "--threshold", type=float, default=0.9, help="similarity threshold")
    parser.add_argument("-s", "--hash-size", type=int, default=16, help="hash size to use, signature length = hash_size^2", dest="hash_size")
    parser.add_argument("-b", "--bands", type=int, default=16, help="number of bands")
    parser.add_argument("-r", "--randomseed", type=int, default=100, help="ensure the same result for the same input directory and threshold, default 100")
    parser.add_argument("-l", "--locality", type=int, default=0, help="0 - do not save the locality, 1 - save the locality, default 0")

    args = parser.parse_args()
    input_dir = args.inputdir
    threshold = args.threshold
    hash_size = args.hash_size
    bands = args.bands
    random_seed = args.randomseed
    save_locality = args.locality

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # creat save path if not exist
    save_path = join(input_dir, "highlight")
    # if folder not exist, create it
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        # clean the folder
        for f in listdir(save_path):
            os.remove(join(save_path, f))

    try:
        near_duplicates, locality = find_near_duplicates(input_dir, threshold, hash_size, bands)
        if near_duplicates:
            print(f"Found {len(near_duplicates)} near-duplicate images in {input_dir} (threshold {threshold:.2%})")
            pair_num = 0
            for a,b,s in near_duplicates:
                pair_num += 1
                print(f"{s:.2%} similarity: file 1: {a} - file 2: {b}")
                if locality[(a,b)] and s < 1.0 and s > 0.97 and save_locality == 1:
                    locality_row, locality_col = locality[(a,b)]
                    # print the mask if the similarity is high but not 100%
                    mask = locality2mask(locality_row, locality_col, bands, int(hash_size**2/bands))
                    # print(mask)
                    # save mask to file
                    np.save(join(save_path, a.split('/')[-1].split('.')[0] + f"_mask_{pair_num}.npy"), mask)
                    img_0, img_1 = highlight_difference(a, b, locality_row, locality_col, bands, int(hash_size**2/bands))
                    img_0.save(join(save_path, a.split('/')[-1].split('.')[0] + f"_highlight_{pair_num}.png"))
                    img_1.save(join(save_path, b.split('/')[-1].split('.')[0] + f"_highlight_{pair_num}.png"))
            #print the number of unique images
            unique_images = len(set([a for a,b,s in near_duplicates] + [b for a,b,s in near_duplicates]))
            print(f"Unique images: {unique_images}")
        else:
            print(f"No near-duplicates found in {input_dir} (threshold {threshold:.2%})")
    except OSError:
        print(f"Couldn't open input directory {input_dir}")
                    

if __name__ == "__main__":
    main(sys.argv)
