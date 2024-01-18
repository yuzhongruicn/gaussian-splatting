import numpy as np
from utils.graphics_utils import rotationMatrixToEulerAngles, radian2angle, angle2radian, eulerAngles2rotationMat


def generate_random_poses_kitti360(cameras, num_frames=100):
    poses = []
    #TODO: generate random poses
    for idx, camera in enumerate(cameras[:-1]):
        w2c = np.eye(4)
        w2c[:3, :3] = camera.R.transpose()
        w2c[:3, 3] = camera.T
        c2w = np.linalg.inv(w2c)
        
        # w2c_next = np.eye(4)
        # camera_next = cameras[idx+1]
        # w2c_next[:3, :3] = camera_next.R.T
        # w2c_next[:3, 3] = camera_next.T
        # c2w_next = np.linalg.inv(w2c_next)
        
        c2w_new = np.eye(4)
        # c2w_new[:3, 3] = 0.5 * (c2w_next[:3, 3] + c2w[:3, 3])
        c2w_new[:3, 3] = c2w[:3, 3]
        euler = rotationMatrixToEulerAngles(c2w[:3, :3])
        angle = radian2angle(euler)
        
        #（-45 - 45）
        angle[2] += (2 * np.random.random() - 1) * 45
        rot = eulerAngles2rotationMat(angle2radian(angle))
        
        c2w_new[:3, :3] = rot
        w2c_new = np.linalg.inv(c2w_new)
        poses.append(w2c_new)

        
    return poses