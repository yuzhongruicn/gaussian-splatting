# source $HOME/miniconda3/bin/activate
# conda activate gaussian_splatting

export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting_handover"

MASK_PATH="/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218/mask"
blocks=(block_0 block_1 block_2 block_3 block_4 block_5 block_6 block_7 block_8 block_9 block_10 block_11 block_12 block_13)

for block in "${blocks[@]}"
do
    SOURCE_PATH="/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218_${block}"
    MODEL_PATH="/root/paddlejob/workspace/yuzhongrui/outputs/3dgs/IDG_ARCF1274_1218/${block}"
    python render.py -s ${SOURCE_PATH} -m ${MODEL_PATH} --skip_train
    python metrics.py -m ${MODEL_PATH} \
                        --mask \
                        --mask_path ${MASK_PATH}
done