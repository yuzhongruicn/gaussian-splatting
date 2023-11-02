source $HOME/miniconda3/bin/activate
conda activate gaussian_splatting

export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting"

args=(
    -s /root/paddlejob/workspace/yuzhongrui/datasets/IDG_Wuhan_1101_block4     # data path
    -m /root/paddlejob/workspace/yuzhongrui/outputs/3dg/IDG_Wuhan_1101_block_4/init     # output / model path

    --resolution 2      # 1/2 resolution
    --eval
    --data_format idg
    # --mask
    --block block_4

    --iterations 50000 
    --test_iterations 7000 20000 30000 40000 50000
    --save_iterations 30000 40000 50000
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