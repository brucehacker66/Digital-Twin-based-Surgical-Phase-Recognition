CUDA_VISIBLE_DEVICES=0 python downstream_phase/run_dt.py \
--batch_size 4 \
--epochs 20 \
--save_ckpt_freq 5 \
--model surgformer_HTA_KCA_dt \
--pretrained_path pretrain_params/TimeSformer_divST_8x32_224_K400.pyth \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /mnt/disk0/haoding/cholec80_original \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--data_set Cholec80 \
--data_fps 1fps \
--output_dir /mnt/disk0/haoding/Surgformer/10mask_1depth_sam/results \
--log_dir /mnt/disk0/haoding/Surgformer/10mask_1depth_sam/results \
--num_workers 12 \
--enable_deepspeed \
--no_auto_resume 
# --dist_eval \