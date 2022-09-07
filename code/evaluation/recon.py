import open3d as o3d
import open3d.core as o3c
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_dir')
    parser.add_argument('--voxel_size', type=float, default=0.015)
    parser.add_argument('--camera_npz', required=True)
    args = parser.parse_args()

    cameras = np.load(args.camera_npz)
    scale = cameras['scale']
    print(scale)
    intrinsic = cameras['intrinsic']

    eval_dir = Path(args.eval_dir)
    rgbs = sorted(eval_dir.glob('*_rgb.png'))
    depths = sorted(eval_dir.glob('*_depth.npy'))
    normals = sorted(eval_dir.glob('*_normal.npy'))
    device = o3c.Device('CUDA:0')
    depth_scale = 1000.0
    depth_max = 5.0

    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'),
                                        attr_dtypes=(o3c.float32, o3c.float32,
                                                     o3c.float32),
                                        attr_channels=((1), (1), (3)),
                                        voxel_size=args.voxel_size,
                                        block_resolution=16,
                                        block_count=50000,
                                        device=device)

    n_files = len(rgbs)
    assert n_files == len(depths)

    for i in tqdm(range(n_files)):
        pose = cameras['pose_%d' % i]
        depth = np.load(depths[i]).squeeze(0) * (1000.0 * scale[0, 0])
        depth = o3d.t.geometry.Image(depth).to(device)

        extrinsic = o3c.Tensor(np.linalg.inv(pose))

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, o3c.Tensor(intrinsic), extrinsic, depth_scale,
            depth_max)

        image = o3d.t.io.read_image(str(rgbs[i])).to(device)
        image = o3d.t.geometry.Image(image.as_tensor().to(o3c.float32) / 255.0)

        vbg.integrate(frustum_block_coords, depth, image,
                      o3c.Tensor(intrinsic), o3c.Tensor(intrinsic),
                      extrinsic, depth_scale, depth_max)

    pcd = vbg.extract_point_cloud(0.0)
    o3d.visualization.draw(pcd)
    o3d.io.write_point_cloud(str(eval_dir / 'pcd.ply'), pcd.to_legacy())
