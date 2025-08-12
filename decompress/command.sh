CUDA_VISIBLE_DEVICES=1 bash run.sh cahq_16_128 train 1 

CUDA_VISIBLE_DEVICES=1 bash run.sh deepgreen_16_256 train 1 

CUDA_VISIBLE_DEVICES=2 bash run.sh green_16_256 train 1 

nohup bash run.sh deepgreen_16_256 train 1 > output_large.log 2>&1 &

bash run.sh ca_16_128 test 1 

CUDA_VISIBLE_DEVICES=4,5,6, bash run.sh deepred_13n_16_256 train 1 
CUDA_VISIBLE_DEVICES=1,2,3 bash run.sh deepred_13n_2_32_256 train 1 

CUDA_VISIBLE_DEVICES=4,5 nohup bash run.sh multisp_all_red_16_256 train 1 > multisp.log 2>&1 &

nohup CUDA_VISIBLE_DEVICES=1 bash run.sh green_16_128 train 1 

nohup bash run.sh green_16_256 train 1 > output_large_1.log 2>&1 &

nohup bash run.sh deepgreen_16_128 train 1 > output0.log 2>&1 &

nohup bash run.sh deepgreensmall_16_128 train 1 > output1.log 2>&1 &

nohup bash run.sh deepredsmall_16_128 train 1 > output2.log 2>&1 &

nohup bash run.sh deepredsmall_32_128 train 1 > output3.log 2>&1 &

nohup CUDA_VISIBLE_DEVICES=1 bash run.sh deepgreen_16_256 train 1  > output0.log 2>&1 &
nohup CUDA_VISIBLE_DEVICES=2 bash run.sh green_16_256 train 1   > output1.log 2>&1 &
# run ca 16 128 as image input
nohup bash run.sh ca_16_128 train 1 &

##########################
#### Data Preparation ####
##########################

# Crop the original images 256x256

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/CA256/green_basecr  --out ../green --size 16,256 

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/CA256/deepgreen_basecr  --out ../deepgreen --size 16,256 

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/CA256/deepgreen_basecr  --out ../deepgreen --size 16,128

# Crop the original images 128x128

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/CA128/deepgreen_small  --out ../deepgreensmall --size 16,128

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/CA128/deepred_small  --out ../deepredsmall --size 16,128

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/CA128/deepred_small  --out ../deepredsmall --size 32,128

# Crop the original images 256x256, for DynamicEarthNet

#Raw data at 256x256

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/DY256_13N/deepgreen --out ../deepgreen_13n --size 16,256

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/DY256_13N/deepred --out ../deepred_13n --size 16,256

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/DY256_13N/deepred --out ../deepred_13n_2 --size 32,256

#Raw data at 128x128

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/DY128_13N/deepgreen --out ../green_13n_small --size 16,128

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/CA/cropped/DY128_13N/deepred --out ../red_13n_small --size 16,128

# Crop the original images 256x256, for DynamicEarthNet multi-spectral (6:9) data

python datasets_prep/prepare_data.py  --path /mnt/raid0sata1/chuanhao/DynamicEarthNet/sentinel2/processed_images_93 --out ../multisp_all_red --size 16,256