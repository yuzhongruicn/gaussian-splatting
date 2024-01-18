#
DATA_PATH='/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1247_0108'

for block_id in {0..4}
do
    block="block_${block_id}"
	path="/root/paddlejob/workspace/yuzhongrui/datasets/IDG_ARCF1274_1218_${block}"
	
    rm ${path}/json
    rm ${path}/mask

	ln -s ${DATA_PATH}/json ${path}/json
	ln -s ${DATA_PATH}/mask ${path}/mask
done
