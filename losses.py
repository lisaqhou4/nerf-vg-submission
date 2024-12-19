import torch
from torch import nn


class NerfWLoss(nn.Module):
    """
    Combined Loss for NeRF-VG
    Adopted from the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
    """
    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets):
        ret = {}
        ret['c_l'] = 0.5 * ((inputs['rgb_coarse']-targets)**2).mean()
        if 'rgb_fine' in inputs:
            ret['f_l'] = 0.5 * ((inputs['rgb_fine']-targets)**2).mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

loss_dict = {'nerfw': NerfWLoss}