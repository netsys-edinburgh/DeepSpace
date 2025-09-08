## Get the data ready ##
Please run 'creat_input.py' to obtain the input data for training DeepSpace

Replace the 'path' with your own data path:

```
def main():
    # Config
    cr = 100
    method = 'sensing'

    path = '/mnt/raid0sata1/chuanhao/CA/cropped'
    files = loadfiles(path)
```

For example, you can download open source dataset from:

[Dynamic Earth Net Dataset](https://mediatum.ub.tum.de/1650201) 

```
Toker, Aysim, et al. "Dynamicearthnet: Daily multi-spectral satellite dataset for semantic change segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
```

## Quick Start ##

You don't need to run this part if you use the following prepared mini dataset:

Please find the mini processed dataset on [Google Drive](https://drive.google.com/drive/folders/15k_WgA8qqc4pFRkS0FEPnAhyhyfEQb18?usp=sharing)

Note that this is not the full size dataset used in our work please refer to the paper for official access.