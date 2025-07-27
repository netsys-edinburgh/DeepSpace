#!/bin/sh
export MASTER_PORT=6002

echo MASTER_PORT=${MASTER_PORT}

export PYTHONPATH=$(pwd):$PYTHONPATH

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=$1
MODE=$2
GPUS=$3

if [ -z "$1" ]; then
   GPUS=1
fi

echo $DATASET $MODE $GPUS

# ----------------- Wavelet -----------
if [[ $MODE == train ]]; then
	echo "==> Training SRWaveDiff"

	if [[ $DATASET == celebahq_16_128 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset celebahq_16_128 --image_size 128 --exp celebahq16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/celebahq_16_128/ \
			--master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 128 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == cahq_16_128 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset cahq_16_128 --image_size 128 --exp cahq16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/cahq_16_128/ \
			--master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 128 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == ca_16_128 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset ca_16_128 --image_size 128 --exp ca16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/ca_16_128/ \
			--master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 128 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == green_16_128 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset green_16_128 --image_size 128 --exp green16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/green_16_128/ \
			--master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 128 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == green_16_256 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset green_16_256 --image_size 256 --exp green16_256_wolatent_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 24 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/green_16_256/ \
			--local_rank 4 --master_port $MASTER_PORT \
			--current_resolution 128 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 256 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == deepgreen_16_256 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset deepgreen_16_256 --image_size 256 --exp deepgreen16_256_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 24 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/deepgreen_16_256/ \
			--num_process_per_node 4 --master_port $MASTER_PORT \
			--current_resolution 128 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 256 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == deepgreen_16_128 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset deepgreen_16_128 --image_size 128 --exp deepgreen16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/deepgreen_16_128/ \
			--node_rank 1 --local_rank 1 --master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 128 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == deepgreensmall_16_128 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset deepgreensmall_16_128 --image_size 128 --exp deepgreensmall16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/deepgreensmall_16_128/ \
			--node_rank 2 --local_rank 2 --master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 128 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == deepredsmall_16_128 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset deepredsmall_16_128 --image_size 128 --exp deepredsmall16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/deepredsmall_16_128/ \
			--node_rank 3 --local_rank 3 --master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 128 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == deepredsmall_32_128 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset deepredsmall_32_128 --image_size 128 --exp deepredsmall32_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 1200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/deepredsmall_32_128/ \
			--master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 32 --h_resolution 128 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == deepred_13n_16_256 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset deepred_13n_16_256 --image_size 256 --exp deepred_13n_wolatent_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 24 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/deepred_13n_16_256 \
			--local_rank 5 --num_process_per_node 3 --master_port $MASTER_PORT \
			--current_resolution 128 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 256 \
			--save_content_every 500 --save_ckpt_every 500 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == deepred_13n_2_32_256 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset deepred_13n_2_32_256 --image_size 256 --exp deepred_13n_2_wolatent_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 24 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/deepred_13n_2_32_256 \
			--local_rank 4 --num_process_per_node 3 --master_port $MASTER_PORT \
			--current_resolution 128 --attn_resolution 32 --num_disc_layers 6 --rec_loss \
			--l_resolution 32 --h_resolution 256 \
			--save_content_every 500 --save_ckpt_every 500 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == multisp_all_red_16_256 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset multisp_all_red_16_256 --image_size 256 --exp multisp_all_red_16_256_wolatent_pywavelets --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 24 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/multisp_all_red_16_256 \
			--local_rank 5 --num_process_per_node 2 --master_port $MASTER_PORT \
			--current_resolution 128 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 256 \
			--save_content_every 500 --save_ckpt_every 500 \
			--use_pytorch_wavelet \

	elif [[ $DATASET == celebahq_16_64 ]]; then #same as celebahq_256 - might need to revisit later
		python train_srwddgan.py --dataset celebahq_16_64 --image_size 64 --exp srwavediff_celebahq_exp3_atn16_wg12224_d5_recloss_500ep --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 128 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir //mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/celebahq_16_64/ \
			--master_port $MASTER_PORT \
			--current_resolution 32 --attn_resolution 16 --num_disc_layers 5 --rec_loss \
			--net_type wavelet \
			--l_resolution 16 --h_resolution 64 \
			

	elif [[ $DATASET == div2k_128_512 ]]; then 
		python train_srwddgan.py --dataset div2k_128_512 --image_size 512 --exp div2k_128_512_batch16_ts4_atn16_wg12224_d6_recloss_200ep --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 2 --num_epoch 200 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/div2k_128_512/ \
			--master_port $MASTER_PORT \
			--current_resolution 256 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 128 --h_resolution 512 \

	elif [[ $DATASET == df2k_128_512 ]]; then 
		python train_srwddgan.py --dataset df2k_128_512 --image_size 512 --exp df2k_128_512_batch2_ts4_atn16_wg112244_d6_pywavelets_recloss_10ep --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 2 --num_epoch 10 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 10 --save_content \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/df2k_128_512/ \
			--master_port $MASTER_PORT \
			--current_resolution 256 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 128 --h_resolution 512 \
			--use_pytorch_wavelet \
			
	fi
else
	echo "==> Testing WaveDiff"

	if [[ $DATASET == celebahq_16_128 ]]; then
		python test_srwddgan.py --dataset celebahq_16_128 --image_size 128 --exp celebahq16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 59 --num_iters 25000 \
			--current_resolution 64 --attn_resolutions 16 \
			--net_type wavelet \
			--l_resolution 16 --h_resolution 128 \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/celebahq_16_128/ \
			--batch_size 64 \
			--compute_fid --real_img_dir ./pytorch_fid/celebahq128_stats.npz \
			#--measure_time \

	elif [[ $DATASET == ca_16_128 ]]; then
		python test_srwddgan.py --dataset ca_16_128 --image_size 128 --exp ca16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_100ep_pywavelets --num_channels 24 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 1170 --num_iters 92500 \
			--current_resolution 64 --attn_resolutions 16 \
			--net_type wavelet \
			--l_resolution 16 --h_resolution 128 \
			--datadir /mnt/raid0sata1/chuanhao/CA/decompress/SRWaveDiff/data/deepgreensmall_16_128 \
			--batch_size 64 \
			--real_img_dir ./pytorch_fid/celebahq128_stats.npz \
			#--measure_time \

	elif [[ $DATASET == celebahq_16_64 ]]; then
		python test_srwddgan.py --dataset celebahq_16_64 --image_size 64 --exp srwavediff_celebahq16-64_exp3_atn16_wg12224_d5_recloss_500ep --num_channels 24 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 100 \
			--current_resolution 32 --attn_resolutions 16 \
			--net_type wavelet \
			--l_resolution 16 --h_resolution 64 \
			--datadir /content/gdrive/MyDrive/srwavediff/datasets/celebahq_16_64/ \
			--batch_size 64 \
			--compute_fid --real_img_dir ./pytorch_fid/celebahq64_stats.npz \
			# --measure_time \

	elif [[ $DATASET == df2k_128_512 ]]; then
		python test_srwddgan.py --dataset df2k_128_512 --image_size 512 --exp df2k_128_512_batch16_ts4_atn16_wg12224_d6_recloss_200ep --num_channels 24 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4 --num_timesteps 4 --num_res_blocks 2  --epoch_id 3 --num_iters 38000   \
			--current_resolution 256 --attn_resolutions 16 \
			--net_type wavelet \
			--l_resolution 128 --h_resolution 512 \
			--datadir /content/gdrive/MyDrive/srwavediff/datasets/df2k_128_512/ \
			--batch_size 2 \
			--compute_fid --real_img_dir ./pytorch_fid/df2k_512_stats.npz \
			# --measure_time \


	fi
fi
