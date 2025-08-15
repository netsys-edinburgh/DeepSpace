<h1 align="center">
  <b>DeepSpace</b><br>
</h1>

# Introduce DeepSpace

This page includes an end2end implementation of DeepSpace.

**DeepSpace**: Super Resolution Powered Efficient and Reliable Satellite Image Data Acquistion

*Chuanhao Sun, Yu Zhang (The University of Edinburgh); Bill Tao, Deepak Vasisht (University of Illinois Urbana-Champaign); Mahesh Marina (The University of Edinburgh)*

**SIGCOMM ’25**, September 8–11, 2025, Coimbra, Portugal

# Install

We use Anaconda to manage the virtual environment.

```bash
conda env create -f environment.yml
conda activate deepspace
```

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

Run `python detect.py -i [your-target-path]` will detet the high similar images in terms of BLSH

## Compress

Besides the BLSH comparison with reference images, the compression in DeepSpace comes with a simple logic - SSIM based resolution selection.

The corresponding code can be founf in '\Compress\creat_input.py'

You must run BLSH with reference images before running the SSIM-based compress.

## Decompress

The decompress process is based on wavelet diffusion, where the SR model for each scenario and resolution shall be trained separately.

To train the model, simply use the `..\decompress\run.sh`

Similarly, `..\decompress\test_srwddgan.py` specifies how to evaluate a trained model.

