import os
import torch
import numpy as np
import glob
import cv2
import utils.general as utils
from utils import rend_util


class SceneDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir,
        img_res,
        scan_id=0,
    ):

        self.instance_dir = data_dir
        print(self.instance_dir)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/omni_image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))

        depth_dir = '{0}/omni_depth'.format(self.instance_dir)

        png_flag = False
        depth_paths = sorted(glob.glob(os.path.join(depth_dir, '*.pfm')))
        if len(depth_paths) != len(image_paths):
            png_flag = True
            depth_paths = sorted(glob.glob(os.path.join(depth_dir, '*.npy')))

        normal_dir = '{0}/omni_normal'.format(self.instance_dir)
        normal_paths = sorted(glob.glob(os.path.join(normal_dir, '*.npy')))

        self.n_images = len(image_paths)
        assert (len(normal_paths) == self.n_images)
        assert (len(depth_paths) == self.n_images)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)

        scale_mat = camera_dict['scale'].astype(np.float32)
        intrinsic_mat = camera_dict['intrinsic'].astype(np.float32)
        pose_mats = [
            camera_dict['pose_%d' % idx].astype(np.float32)
            for idx in range(self.n_images)
        ]

        self.intrinsic = torch.from_numpy(intrinsic_mat)
        self.pose_all = []

        inv_scale = np.linalg.inv(scale_mat).astype(np.float32)
        for pose in pose_mats:
            self.pose_all.append(torch.from_numpy(inv_scale @ pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.depth_images = []
        for path in depth_paths:
            depth = np.load(path)
            depth = depth.flatten().astype(np.float32)
            self.depth_images.append(
                torch.from_numpy(depth).float().unsqueeze(-1))

        self.normal_images = []
        for i, path in enumerate(normal_paths):
            normal = np.load(path)
            normal = normal.reshape(3, -1).transpose(1, 0)
            normal = (normal - 0.5) * 2
            normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = normal / normal_norm

            # Camera to world
            R = self.pose_all[i][:3, :3].numpy()
            normal = normal @ R.T

            # import matplotlib.pyplot as plt
            # plt.imshow(np.abs(normal.reshape((384, 384, 3))))
            # plt.show()

            self.normal_images.append(torch.from_numpy(normal).float())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsic,
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx]
            ground_truth["normal"] = self.normal_images[idx][
                self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(
                self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale']
