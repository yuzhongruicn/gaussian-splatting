
import numpy as np
import os
import cv2
import json

import argparse

from copy import deepcopy


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

if __name__ == '__main__':
    source_path = '/root/paddlejob/workspace/yuzhongrui/datasets/nuScenes/trainval_results_5/dense/train'
    target_path = '/root/paddlejob/workspace/yuzhongrui/datasets/nuScenes/trainval_results_5/images/train'
    json_path = '/root/paddlejob/workspace/yuzhongrui/datasets/nuScenes/trainval_results_5'
    
    prompt_json_path = os.path.join(json_path, 'prompt.json')
    promt_json_list = []
    
    source_list = [x for x in os.listdir(source_path) if is_image_file(x)]
    source_list.sort()
    print(len(source_list))
    
    for name in source_list:
        scene_id = int(name.split('_')[0].split('-')[-1])
        if scene_id > 991:
            continue
        promt_dict = {}
        promt_dict["source"] = os.path.join(source_path, name)
        promt_dict["target"] = os.path.join(target_path, name)
        promt_dict["prompt"] = "realistic streetview"
        promt_json_list.append(promt_dict)  
        
    with open(prompt_json_path, 'w') as f:
        for entry in promt_json_list:
            f.write(json.dumps(entry) + '\n')