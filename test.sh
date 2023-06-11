CUDA_VISIBLE_DEVICES=2 nohup python /data/home/huangshan/superResolution/mscmr1/test_demo.py \
--scale 4 --dataset mri --model_type gfpgan \
--batch_size 2 \
--net_G gfpgan \
--resume /data/home/huangshan/superResolution/mscmr/outputs/gfpgan_ixi_4_axial/checkpoints/best_107.pt \
--experiment_name gfpgan_ixi_4_axial --target_modal t2 \
--data_root /data/home/share_data/huangshan_16022/dataset/ixi/resample1 --pretrained_path /data/home/share_data/huangshan_16022/pretrained/gfpgan_iter_495000.pth \
--embed_dim 60 >./out/gfpgan_ixi_4_axial_t2.out  2>&1 &

/data/home/huangshan/superResolution/mscmr1/outputs/minet_ixi_4_axial/checkpoints/best_15.pt

CUDA_VISIBLE_DEVICES=0 nohup python /data/home/huangshan/superResolution/mscmr1/test_demo.py \
--scale 4 --dataset mri --model_type minet \
--batch_size 6 \
--net_G minet \
--resume /data/home/huangshan/superResolution/mscmr1/outputs/minet_ixi_4_axial/checkpoints/best_63.pt \
--experiment_name minet_ixi_4_axial --target_modal t2 \
--data_root /data/home/share_data/huangshan_16022/dataset/ixi/resample1 --pretrained_path /home3/huangshan/superResolution/mscmr1/pretrained/gfpgan_iter_495000.pth \
--embed_dim 60 >./minet_ixi_4_axial.out  2>&1 &

#tgp
CUDA_VISIBLE_DEVICES=0 nohup python /data/home/huangshan/superResolution/mscmr1/test_demo.py \
--scale 8 --dataset mri --model_type tgp7 \
--batch_size 4 \
--net_G tgp7 \
--resume /data/home/huangshan/superResolution/mscmr1/outputs/transmr_t1_8_center_tgp7/checkpoints/best_11.pt \
--experiment_name transmrsr_ixi_8_t2_center_test_1_fix --target_modal t1 \
--data_root /data/home/share_data/huangshan_16022/dataset/ixi/resample1 --pretrained_path /data/home/share_data/huangshan_16022/styleswin_900000.pt \
--truncation_path /data/home/huangshan/superResolution/mscmr1/out/clusters/00049-multimodal-truncation-8clusters-pure_centroids \
--embed_dim 60 >./out/transmrsr_ixi_8_t1_center_test1.out  2>&1 &

#swinir 4
CUDA_VISIBLE_DEVICES=2 nohup python /data/home/huangshan/superResolution/mscmr1/test_demo.py \
--scale 4 --dataset mri --model_type swinir \
--batch_size 6 \
--net_G swinir \
--resume /data/home/huangshan/superResolution/mscmr1/outputs/swinir_ixi_4_axial_60/checkpoints/best_47.pt \
--experiment_name swinir_ixi_4_t2t1 --target_modal t2 \
--data_root /data/home/share_data/huangshan_16022/dataset/ixi/resample1 --pretrained_path /data/home/huangshan/superResolution/mscmr1/styleswin_900000.pt \
--embed_dim 60 >./swinir_ixi_4_axial_t2-t1.out  2>&1 &


#tgp7
CUDA_VISIBLE_DEVICES=2 nohup python /data/home/huangshan/superResolution/mscmr1/test_demo.py \
--scale 4 --model_type tgp7 --dataset real \
--batch_size 12 \
--net_G tgp7 \
--resume /data/home/huangshan/superResolution/mscmr1/outputs/transmr_t1_8_center_tgp7/checkpoints/best_11.pt \
--experiment_name transmrsr_ixi_8_t1_center_test_1_tgp7_test_hualiao --target_modal t1 \
--data_root /data/home/share_data/huangshan_16022/dataset/hualiao --pretrained_path /data/home/share_data/huangshan_16022/styleswin_900000.pt \
--truncation_path /data/home/huangshan/superResolution/mscmr1/out/clusters/00049-multimodal-truncation-8clusters-pure_centroids \
--embed_dim 60 >./out/transmrsr_ixi_8_t1_center_test_1_tgp7_test_hualiao.out 2>&1 &


##tgp6
CUDA_VISIBLE_DEVICES=2 nohup python /data/home/huangshan/superResolution/mscmr1/test_demo.py \
--scale 8 --dataset mri --model_type tgp7 \
--batch_size 4 \
--net_G tgp6 \
--resume /data/home/huangshan/superResolution/mscmr1/outputs/transmr_t1_8_center_tgp4_fixdecoder/checkpoints/best_43.pt \
--experiment_name transmrsr_ixi_8_t1_center_test_1_tgp6_test --target_modal t1 \
--data_root /data/home/share_data/huangshan_16022/dataset/ixi/resample1 --pretrained_path /data/home/share_data/huangshan_16022/styleswin_900000.pt \
--truncation_path /data/home/huangshan/superResolution/mscmr1/out/clusters/00049-multimodal-truncation-8clusters-pure_centroids \
--embed_dim 60 >./out/transmrsr_ixi_8_t1_center_test_1_tgp6_test.out  2>&1 &

/data/home/huangshan/superResolution/mscmr1/outputs/transmr_t1_8_center_tgp7_wogp/checkpoints/best_23.pt
CUDA_VISIBLE_DEVICES=1 nohup python /data/home/huangshan/superResolution/mscmr1/test_demo.py \
--scale 8 --model_type tgp7 --dataset mri \
--batch_size 12 \
--net_G tgp7 \
--resume /data/home/huangshan/superResolution/mscmr1/outputs/transmr_t1_8_center_tgp7_wogp-3/checkpoints/model_11.pt \
--experiment_name transmrsr_ixi_8_t1_center_test_1_tgp7_test_ixi_wogp3 --target_modal t1 \
--data_root /data/home/share_data/huangshan_16022/dataset/ixi/resample1 \
--embed_dim 60 >./out/transmrsr_ixi_8_t1_center_test_1_tgp7_test_ixi_wogp3.out 2>&1 &