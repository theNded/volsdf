import os
import glob
import argparse
import numpy as np

# For Manhattan SDF dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene_dir')
    args = parser.parse_args()

    intrinsic = np.loadtxt(os.path.join(args.scene_dir,
                                        'intrinsic.txt'))[:3, :3]
    extrinsics = np.loadtxt(os.path.join(args.scene_dir,
                                         'trajectory.txt')).reshape((-1, 4, 4))

    camera_dict = {}
    for i, extrinsic in enumerate(extrinsics):
        P = intrinsic @ extrinsic[:3]
        camera_dict['world_mat_%d' % i] = P

    np.savez(os.path.join(args.scene_dir, 'cameras_unormalized.npz'),
             **camera_dict)
