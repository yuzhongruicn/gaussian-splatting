#
DATA_PATH='/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218'
blocks=(block_0 block_1 block_2 block_3 block_4 block_5 block_6 block_7 block_8 block_9 block_10 block_11 block_12 block_13)

for block in "${blocks[@]}"
do
	path="/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218_${block}"
	
    rm ${path}/json
    rm ${path}/mask

	ln -s ${DATA_PATH}/json ${path}/json
	ln -s ${DATA_PATH}/mask ${path}/mask
done
