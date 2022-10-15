import open3d as o3d
import open3d.core as o3c
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--intrinsic", required=True)
    args = parser.parse_args()
    eval_subdir = Path(args.data_dir)
    rgbs = sorted(eval_subdir.glob("eval_*.png"))
    depths = sorted(eval_subdir.glob("depth_*.npy"))

    print(depths)
    intrinsic = np.loadtxt(args.intrinsic)
    poses = np.loadtxt(args.poses).reshape((-1, 4, 4))

    device = o3c.Device("CUDA:0")
    depth_scale = 1000.0
    depth_max = 3.0

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=args.voxel_size,
        block_resolution=16,
        block_count=50000,
        device=device,
    )

    n_files = len(rgbs)
    assert n_files == len(depths)

    for i in tqdm(range(n_files)):
        print(i, depths[i])
        pose = poses[i]
        depth = np.load(depths[i]) * 1000.0

        # import matplotlib.pyplot as plt
        # plt.imshow(depth)
        # plt.show()
        depth = o3d.t.geometry.Image(depth).to(device)

        extrinsic = o3c.Tensor(np.linalg.inv(pose))

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, o3c.Tensor(intrinsic), extrinsic, depth_scale, depth_max
        )

        image = o3d.t.io.read_image(str(rgbs[i])).to(device)
        image = o3d.t.geometry.Image(image.as_tensor().to(o3c.float32) / 255.0)

        vbg.integrate(
            frustum_block_coords,
            depth,
            image,
            o3c.Tensor(intrinsic),
            o3c.Tensor(intrinsic),
            extrinsic,
            depth_scale,
            depth_max,
        )

    pcd = vbg.extract_point_cloud(3.0)
    o3d.visualization.draw(pcd)
    o3d.io.write_point_cloud(
        'test.ply',
        # str(eval_rendering_dir / "tsdf_refusion_{}.ply".format(eval_subdir.stem)),
        pcd.to_legacy(),
    )
