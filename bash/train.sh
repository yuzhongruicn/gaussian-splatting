source $HOME/miniconda3/bin/activate
conda activate gaussian_splatting

export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting"

args=(
    -s /root/paddlejob/workspace/yuzhongrui/datasets/IDG_Wuhan_0907_block0       # data path
    -m /root/paddlejob/workspace/yuzhongrui/outputs/IDG_Wuhan_block_0/init_100000     # output / model path

    --resolution 2      # 1/2 resolution
    --eval
    --data_format idg

    --iterations 100000 
    --test_iterations 7000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
    --save_iterations 60000 70000 80000 90000 100000
    )

python train.py "${args[@]}"