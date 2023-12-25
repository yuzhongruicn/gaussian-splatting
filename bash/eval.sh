# source $HOME/miniconda3/bin/activate
# conda activate gaussian_splatting

export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting_handover"

SOURCE_PATH='/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218/mask'
MODEL_PATH="/root/paddlejob/workspace/yuzhongrui/outputs/3dgs/IDG_ARCF1274_1218/block_12"

python render.py -m ${MODEL_PATH} --skip_train
python metrics.py -m ${MODEL_PATH} \
                    --mask \
                    --mask_path ${SOURCE_PATH}