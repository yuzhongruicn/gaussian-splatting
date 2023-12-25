
export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting_handover"


args=(
    --data_path /root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218
    --all_model_path /root/paddlejob/workspace/yuzhongrui/outputs/3dgs/IDG_ARCF1274_1218/
    --iteration 50000
    )


python render_novelviews.py "${args[@]}"