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
    -m /root/paddlejob/workspace/yuzhongrui/outputs/3dg/IDG_ARCF1332_1117_block1/d_sh_xyz_embed_init_lr_10_D8_W256_bpsh # output / model path

    # # wandb params
    # --wandb_disabled
    --project_name "3D Gaussian Splatting"
    --run_name "appearance_embedding/d_sh_xyz_embed_init_lr_10_D8_W256_bpsh"

    --resolution 2      # 1/2 resolution
    --eval
    --data_format idg
    # --mask
    --block block_1

    --load2gpu_on_the_fly

    --iterations 50000 
    --test_iterations 10 7000 10000 20000 30000 40000 50000
    --save_iterations 50000
   
    # # for debug
    # --wandb_disabled
    # --iterations 100
    # --test_iterations 50
    # --warm_up 10
     # --convert_SHs_python

    # # optimization params
    --densify_until_iter 35000
   
    --position_lr_init 0.00960             # default 0.00016
    --position_lr_final 0.0000016          # default 0.0000016
    --position_lr_max_steps 50000

    --ap_lr_init 0.0010
    --ap_lr_final 0.00001
    --ap_lr_max_steps 50_000
    --warm_up 5000

    --ap_num_hidden_layers 8
    --ap_num_hidden_neurons 256

    # # add spherical bg
    # --spherical_bg
    # --num_bg_points 20000
    # --bg_dist 3.0

    )


python train_ape.py "${args[@]}"