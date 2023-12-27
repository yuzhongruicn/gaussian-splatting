#
export PYTHONPATH="${PYTHONPATH}:/root/paddlejob/workspace/yuzhongrui/project/gaussian-splatting_handover"
DATA_PATH='/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218'
blocks=(block_0 block_1)

for block in "${blocks[@]}"
do
	path="/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218_${block}"
	python tools/idg_to_colmap.py --data_path ${DATA_PATH} --output_path ${path} --block ${block} --downsample 1

	colmap feature_extractor --database_path ${path}/database.db --image_path ${path}/images
	python tools/database.py --data_path ${path}
	python regenerate_imagetxt_idg.py --data_path ${DATA_PATH} --output_path ${path}  --block ${block}
	colmap exhaustive_matcher --database_path ${path}/database.db

	mkdir ${path}/sparse/0
	mkdir ${path}/sparse/2
	touch ${path}/sparse/1/points3D.txt
	colmap point_triangulator --database_path ${path}/database.db --image_path ${path}/images --input_path ${path}/sparse/1 --output_path ${path}/sparse/2
	colmap model_converter --input_path ${path}/sparse/2 --output_path ${path}/sparse/0 --output_type TXT
	
	ln -s ${DATA_PATH}/json ${path}/json
	ln -s ${DATA_PATH}/mask ${path}/mask
done
