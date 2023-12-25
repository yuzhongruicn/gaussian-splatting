# source $HOME/miniconda3/bin/activate
# conda activate gaussian_splatting

export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting_handover"

MASK_PATH="/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218/mask"
blocks=(block_11)

for block in "${blocks[@]}"
do
    MODEL_PATH="/root/paddlejob/workspace/yuzhongrui/outputs/3dgs/IDG_ARCF1274_1218/${block}"
    python render.py -m ${MODEL_PATH} --skip_train
    python metrics.py -m ${MODEL_PATH} \
                        --mask \
                        --mask_path ${MASK_PATH}
done