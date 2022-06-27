import torch
from torch import nn
import utils.general as utils


class VolSDFLoss(nn.Module):

    def __init__(self,
                 rgb_loss,
                 eikonal_weight,
                 depth_weight=0.0,
                 normal_weight=0.00):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.depth_weight = depth_weight
        self.normal_weight = normal_weight

        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_depth_loss(self, depth_values, depth_gt):
        return (depth_values - depth_gt).mean()

    def get_normal_loss(self, normal_values, normal_gt):
        l1_loss = torch.abs(normal_values - normal_gt).mean()
        dot_loss = torch.abs(1 - (normal_values * normal_gt).sum(dim=1)).mean()
        return l1_loss + dot_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1)**2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()
        depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        depth_loss = self.get_depth_loss(model_outputs['depth_values'],
                                         depth_gt)
        normal_loss = self.get_normal_loss(model_outputs['normal_values'],
                                           normal_gt)

        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss +\
               self.depth_weight * depth_loss +\
               self.normal_weight * normal_loss

        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'depth_loss': depth_loss,
            'normal_loss': normal_loss,
            'eikonal_loss': eikonal_loss,
        }

        return output
