import cv2
import numpy as np
import argparse


def get_center_point(intrinsic, poses):
    num_cams = len(poses)

    A = np.zeros((3 * num_cams, 3 + num_cams))
    b = np.zeros((3 * num_cams, ))

    # Solve for x
    # x: (center (3 dim) + radius (n dim))
    for i in range(num_cams):
        # Z-axis of the camera, facing inside the screen
        z_axis = poses[i][:3, 2]
        position = poses[i][:3, 3]

        A[3 * i:(3 * i + 3), :3] = np.eye(3)
        A[3 * i:(3 * i + 3), 3 + i] = -z_axis
        b[3 * i:(3 * i + 3)] = position

    x = np.linalg.lstsq(A, b, rcond=None)

    return x[0], b.reshape((3, num_cams))


def normalize_cameras(intrinsic_dir, poses_dir, output_dir):
    intrinsic = np.loadtxt(intrinsic_dir)
    poses = np.loadtxt(poses_dir).reshape((-1, 4, 4))

    x, camera_centers = get_center_point(intrinsic, poses)
    center = x[:3].flatten()

    max_radius = np.linalg.norm(
        (center[:, np.newaxis] - camera_centers), axis=0).max() * 1.1

    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = center[0]
    normalization[1, 3] = center[1]
    normalization[2, 3] = center[2]

    normalization[0, 0] = max_radius / 3.0
    normalization[1, 1] = max_radius / 3.0
    normalization[2, 2] = max_radius / 3.0

    print(normalization)

    n_cameras = len(poses)
    cameras = {}
    cameras['scale'] = normalization
    cameras['intrinsic'] = intrinsic
    for i in range(n_cameras):
        cameras['pose_%d' % i] = poses[i]
    np.savez(output_dir, **cameras)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalizing cameras')
    parser.add_argument('--intrinsic', type=str, required=True)
    parser.add_argument('--poses', type=str, required=True)
    parser.add_argument('--output', type=str, default='cameras.npz')
    args = parser.parse_args()

    normalize_cameras(args.intrinsic, args.poses, args.output)
