# source $HOME/miniconda3/bin/activate
# conda activate gaussian_splatting

export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting"

############## kitti
# args=(
#     -s /root/paddlejob/workspace/yuzhongrui/datasets/2011_09_26_kitti_original     # data path
#     -m /root/paddlejob/workspace/yuzhongrui/outputs/3dg/2011_09_26_kitti_original/init_original_lr_60_combined_voxel0.5   # output / model path
    
#     # # wandb params
#     # --wandb_disabled
#     # --project_name "3D Gaussian Splatting"
#     --run_name "kitti_110926_init_original_lr_60_combined_voxel0.5"

#     --eval

#     --iterations 30000 
#     --test_iterations 7000 20000 30000
#     --save_iterations 30000

#     --position_lr_init 0.00960             # default 0.00016
#     --position_lr_final 0.0000016          # default 0.0000016
#     --position_lr_max_steps 30000
#     )

############## IDG
args=(
    -s /root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1332_1117_block1/     # data path
    -m /root/paddlejob/workspace/yuzhongrui/outputs/3dg/IDG_ARCF1332_1117_block1/test # output / model path

    # # wandb params
    # --wandb_disabled
    # --project_name "3D Gaussian Splatting"
    # --run_name "1117_colmap_init_mask_detec_lr_init_60"

    --resolution 2      # 1/2 resolution
    --eval
    --data_format idg
    # --mask
    --block block_1

    # --iterations 50000 
    # --test_iterations 7000 20000 30000 40000 50000
    # --save_iterations 50000

    # # for debug
    --wandb_disabled
    --iterations 100
    --test_iterations 50

    # # optimization params
    --densify_until_iter 35000

    --position_lr_init 0.00960             # default 0.00016
    --position_lr_final 0.0000016          # default 0.0000016
    --position_lr_max_steps 50000

    # # add spherical bg
    # --spherical_bg
    # --num_bg_points 20000
    # --bg_dist 3.0

    )


# args=(
#     -s /root/paddlejob/workspace/yuzhongrui/datasets/IDG_Wuhan_0907_block0       # data path
#     -m /root/paddlejob/workspace/yuzhongrui/outputs/IDG_Wuhan_block_0/test     # output / model path

#     --resolution 2      # 1/2 resolution
#     --eval
#     --data_format idg

#     --mask

#     --iterations 100
#     --test_iterations 100
#     )

python train.py "${args[@]}"