CUDA_VISIBLE_DEVICES=0 python downstream_phase/run_phase_training.py \
--batch_size 4 \
--epochs 20 \
--save_ckpt_freq 5 \
--model surgformer_HTA_KCA \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /mnt/disk0/haoding/cholec80_41-50combine \
--eval_data_path /mnt/disk0/haoding/cholec80_41-50combine \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--eval \
--finetune /mnt/disk0/haoding/Surgformer_results/1mask_2depth/results/surgformer_HTA_KCA_Cholec80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-best/mp_rank_00_model_states.pt \
--data_set Cholec80 \
--data_fps 1fps \
--output_dir /mnt/disk0/haoding/Surgformer_results/1mask_2depth/results/41-50_combine \
--log_dir /mnt/disk0/haoding/Surgformer_results/1mask_2depth/results/41-50_combine \
--num_workers 15 \
--channel_type 1mask_2depth \
# --enable_deepspeed \
# --no_auto_resume