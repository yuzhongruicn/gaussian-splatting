# source $HOME/miniconda3/bin/activate
# conda activate gaussian_splatting

export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting_handover"
blocks=(block_11)

for block in "${blocks[@]}"
do
	args=(
        -s /root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218_${block}/     # 数据路径
        -m /root/paddlejob/workspace/yuzhongrui/outputs/3dgs/IDG_ARCF1274_1218/${block}     # 保存模型的路径
        --block ${block}     # 选择block
        
        ##### logger params
        # --logger tensorboard    # 选择用什么logger 可选['wandb', 'tensorboard'], 注释掉这行表示不使用任何logger

        ##### training params
        --resolution 2            # 图像降采样到 1/2 resolution
        --eval                    # 是否分train和val
        --data_format idg         # 数据格式
        --mask                    # 是否使用mask

        --iterations 50000 
        --test_iterations 7000 10000 20000 30000 40000 50000
        --save_iterations 50000

        ##### optimization params
        --densify_until_iter 35000
    
        --position_lr_init 0.00960
        --position_lr_final 0.0000016
        --position_lr_max_steps 50000

        ##### appearance embedding params
        # --add_appearance_embedding
        # --embedding_dim 32
        # --ap_num_hidden_layers 4
        # --ap_num_hidden_neurons 128
        
        # --ap_lr_init 0.0010
        # --ap_lr_final 0.00001
        # --ap_lr_max_steps 50_000
        # --warm_up 5000
        
        # --load2gpu_on_the_fly
        )


    python train.py "${args[@]}"
done

############## IDG
# args=(
#     -s /root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218_block_0/     # 数据路径
#     -m /root/paddlejob/workspace/yuzhongrui/outputs/3dgs/IDG_ARCF1274_1218/block_0     # 保存模型的路径
#     --block block_0     # 选择block
    
#     ##### logger params
#     # --logger tensorboard    # 选择用什么logger 可选['wandb', 'tensorboard'], 注释掉这行表示不使用任何logger

#     ##### training params
#     --resolution 2            # 图像降采样到 1/2 resolution
#     --eval                    # 是否分train和val
#     --data_format idg         # 数据格式
#     --mask                    # 是否使用mask

#     --iterations 50000 
#     --test_iterations 7000 10000 20000 30000 40000 50000
#     --save_iterations 50000

#     ##### optimization params
#     --densify_until_iter 35000
   
#     --position_lr_init 0.00960
#     --position_lr_final 0.0000016
#     --position_lr_max_steps 50000

#     ##### appearance embedding params
#     --add_appearance_embedding
#     --embedding_dim 32
#     --ap_num_hidden_layers 4
#     --ap_num_hidden_neurons 128
    
#     --ap_lr_init 0.0010
#     --ap_lr_final 0.00001
#     --ap_lr_max_steps 50_000
#     --warm_up 5000
    
#     --load2gpu_on_the_fly
#     )


# python train.py "${args[@]}"