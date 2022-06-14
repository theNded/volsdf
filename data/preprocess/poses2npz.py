import os
import glob
import argparse
import numpy as np

# For Manhattan SDF dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_dir')
    parser.add_argument('intrinsic_dir')
    args = parser.parse_args()

    K = np.loadtxt(args.intrinsic_dir)[:3, :3]
    pose_fnames = sorted(glob.glob(os.path.join(args.pose_dir, '*.txt')))

    camera_dict = {}
    for i, pose_fname in enumerate(pose_fnames):
        extrinsic = np.linalg.inv(np.loadtxt(pose_fname))
        P = K @ extrinsic[:3]

        camera_dict['world_mat_%d' % i] = P

    np.savez(os.path.join(args.pose_dir, 'cameras_unormalized.npz'), **camera_dict)
