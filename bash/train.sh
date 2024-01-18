# source $HOME/miniconda3/bin/activate
# conda activate gaussian_splatting

export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting"
############## IDG
args=(
    -s /root/paddlejob/workspace/yuzhongrui/datasets/kitti360/2013_05_28_drive_0009_sync/block_4  # 数据路径
    -m /root/paddlejob/workspace/yuzhongrui/outputs/3dgs/kitti360/05280009_block_4_lidar_pseudoviews_sdsloss_interval100  # 保存模型的路径
    
    ##### logger params
    --logger wandb    # 选择用什么logger 可选['wandb', 'tensorboard'], 注释掉这行表示不使用任何logger
    # --online
    --project_name "3D Gaussian Splatting"
    --run_name "kitti360/05280009_block_4_lidar_sdsloss_interval100"

    ##### training params
    # --resolution 2            # 图像降采样到 1/2 resolution
    --eval                    # 是否分train和val
    --data_format colmap         # 数据格式
    # --mask                    # 是否使用mask

    --iterations 50000 
    --test_iterations 1 7000 10000 20000 30000 40000 50000
    --save_iterations 50000

    ##### optimization params
    --densify_until_iter 35000
   
    --position_lr_init 0.00016
    --position_lr_final 0.0000016
    --position_lr_max_steps 50000

    --enable_sd
    --sd_batch_size 4
    # --start_sample_pseudo 1
    --sample_pseudo_interval 100
    )


python train.py "${args[@]}"