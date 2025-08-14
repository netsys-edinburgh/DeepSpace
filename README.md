<h1 align="center">
  <b>DeepSpace</b><br>
</h1>

# Introduce DeepSpace

This repo includes an end2end implementation of DeepSpace.

DeepSpace: Super Resolution Powered Efficient and Reliable Satellite Image Data Acquistion

SIGCOMM ’25, September 8–11, 2025, Coimbra, Portugal

# Usage 

## Template Dataset

Please find the processed showcase dataset on [Google Drive](https://drive.google.com/drive/folders/15k_WgA8qqc4pFRkS0FEPnAhyhyfEQb18?usp=sharing)

Then unzip the data to `.../decompress/data/`.

This dataset is based on Planet-California, please cite the original data source as

```bash
@article{devaraj2017dove,
  title={Dove high speed downlink system},
  author={Devaraj, Kiruthika and Kingsbury, Ryan and Ligon, Matt and Breu, Joseph and Vittaldev, Vivek and Klofas, Bryan and Yeon, Patrick and Colton, Kyle},
  year={2017}
}
```

The datasets used in this work are open-source, please refer to their original source for further access.

## Similar image detection and LSH encoding process

The code can be found in '\BLSH'

Please ensure ImageHash is available before using BLSH module.

## Compress

Compression in DeepSpace comes with a simple logic - SSIM based resolution selection.
