# plot the coordinates trace of a satellite image series in a folder

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from trace import *

def get_folfer(path, key_word='meta'):
    # return all folder include meta in name
    return [f for f in os.listdir(path) if f.find(key_word) != -1]

# data_dir = '/Users/sunchuanhao/code/myprocess/farmvibe/7_15'

# coordination = load_trace(data_dir)
# plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='trace_farm.pdf')

# data_dir = '/Users/sunchuanhao/code/myprocess/Not_CA_for_tranining/1f420bf4-5ad1-44bf-9d18-bad31d81d8c7/PSScene'

# coordination = load_trace(data_dir)
# plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='trace_ca_0.pdf')

# data_dir = '/Users/sunchuanhao/code/myprocess/Not_CA_for_tranining/1645fd15-84ab-4be0-9870-1abb9cda2614/PSScene'

# coordination = load_trace(data_dir)
# plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='trace_ca_1.pdf')
# plot_coverage(coordination, save_path='cover_ca_1.pdf')

# data_dir = '/Users/sunchuanhao/code/myprocess/farmvibe/7_15_2/e646ce87-0b6c-4949-86ad-14415d38d00f/PSScene'

# coordination = load_trace(data_dir)
# plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='farm_15_2.pdf')

# data_dir = '/Users/sunchuanhao/code/myprocess/farmvibe/7_19'

# coordination = load_trace(data_dir)
# # plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='farm_19_0.pdf')
# plot_coverage(coordination, save_path='cover_farm_19_0.pdf')

# data_dir = '/Users/sunchuanhao/code/myprocess/farmvibe/'

# folders = get_folfer(data_dir, key_word='7_15_meta')

# for folder in folders:
#     coordination = load_trace(os.path.join(data_dir, folder))
#     plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='imgs/farm_7_15/farm_15_{}.pdf'.format(folder.split('_')[-1]))
#     plot_coverage(coordination, mode='nipy_spectral', save_path='imgs/farm_7_15/cover_farm_15_{}.pdf'.format(folder.split('_')[-1]))



# data_dir = '/Users/sunchuanhao/code/myprocess/farmvibe/'

# folders = get_folfer(data_dir, key_word='7_15_meta')

# for idx, folder in enumerate(folders):
#     if idx == 0:
#         coordination = load_trace(os.path.join(data_dir, folder))
#     else:
#         coordination += load_trace(os.path.join(data_dir, folder))

# plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='imgs/farm_7_15/farm_15_all.pdf')
# plot_coverage(coordination, mode='nipy_spectral', save_path='imgs/farm_7_15/cover_farm_15_all.pdf')



# data_dir = '/Users/sunchuanhao/code/myprocess/farmvibe/'

# folders = get_folfer(data_dir, key_word='7_19_meta')

# for idx, folder in enumerate(folders):
#     if idx == 0:
#         coordination = load_trace(os.path.join(data_dir, folder))
#     else:
#         coordination += load_trace(os.path.join(data_dir, folder))
# plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='imgs/farm_7_19/farm_19_all.pdf')
# plot_coverage(coordination, mode='nipy_spectral', save_path='imgs/farm_7_19/cover_farm_19_all.pdf')

# for folder in folders:
#     coordination = load_trace(os.path.join(data_dir, folder))
#     plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='imgs/farm_7_19/farm_19_{}.pdf'.format(folder.split('_')[-1]))
#     plot_coverage(coordination, mode='nipy_spectral', save_path='imgs/farm_7_19/cover_farm_19_{}.pdf'.format(folder.split('_')[-1]))


# data_dir = '/Users/sunchuanhao/code/myprocess/Not_CA_for_training/'

# folders = get_folfer(data_dir, key_word='ca_meta')

# for folder in folders:
#     coordination = load_trace(os.path.join(data_dir, folder))
#     plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path='imgs/ca_15days/ca_{}.pdf'.format(folder.split('_')[-1]))
#     plot_coverage(coordination, mode='nipy_spectral', save_path='imgs/ca_15days/cover_ca_{}.pdf'.format(folder.split('_')[-1]))

data_dir = '/Users/sunchuanhao/code/myprocess/Not_CA_for_training/'

folders = get_folfer(data_dir, key_word='ca_meta')
# sort folder by the last number (check one digit or two digit)
folders.sort(key=lambda x: int(x.split('_')[-1]))
pair = [52, 68, 26, 27, 40, 51]
count = 0
for idx, folder in enumerate(folders):
    if int(folder.split('_')[-1]) not in pair:
        continue
    else:
       count += 1
       if count == 1:
           coordination = load_trace(os.path.join(data_dir, folder))
       else:
              coordination += load_trace(os.path.join(data_dir, folder))
plot_trace(coordination, separate_neighbours=True, mode='nipy_spectral', save_path=f'imgs/ca_15days/ca_{pair[0]}_{pair[1]}.pdf')
plot_coverage(coordination, mode='nipy_spectral', save_path=f'imgs/ca_15days/cover_ca_{pair[0]}_{pair[1]}.pdf')