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
import os

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import CIFAR10, STL10

from .LRHR_dataset import LRHRDataset

def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    elif dataset == 'ffhq':
        return 63000 if train else 7000
    elif dataset == 'cahq':
        return 5000 if train else 600
    elif dataset == 'dyred':
        return 1000 if train else 600
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


def create_dataset(args):
    if args.dataset == 'celebahq_16_64':
        dataset = LRHRDataset(
                dataroot=args.datadir,
                datatype='lmdb',
                l_resolution=args.l_resolution,
                r_resolution=args.h_resolution,
                split="train",
                data_len=-1,
                need_LR=False
                )
        
    elif args.dataset == 'celebahq_16_128':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='lmdb',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
     
    # cahq_16_128 and ca_16_128 are the same dataset in different formats    
    # ca_16_128 is the image format, means the dataset is stored as images
    elif args.dataset == 'cahq_16_128':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='lmdb',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'ca_16_128':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'green_16_128':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'green_16_256':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'deepgreen_16_256':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'deepgreen_16_128':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'deepgreensmall_16_128':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'deepredsmall_16_128':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'deepredsmall_32_128':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'deepred_13n_16_256':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'deepred_13n_2_32_256':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'multisp_all_red_16_256':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='img',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
        
    elif args.dataset == 'div2k_128_512':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='lmdb',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
    elif args.dataset == 'df2k_128_512':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='lmdb',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )


    return dataset
