import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T

from .ray_utils import *



class PersonDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800),
                 perturbation=[]):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_train.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        fl_x = self.meta.get('fl_x')
        fl_y = self.meta.get('fl_y')
        cx = self.meta.get('cx', w/2)
        cy = self.meta.get('cy', h/2)

        self.K = np.eye(3)
        self.K[0, 0] = fl_x * (self.img_wh[0] / self.meta["w"])  # Scale focal length to match img_wh
        self.K[1, 1] = fl_y * (self.img_wh[1] / self.meta["h"])  # Scale focal length to match img_wh
        self.K[0, 2] = cx * (self.img_wh[0] / self.meta["w"])  # Scale cx to match img_wh
        self.K[1, 2] = cy * (self.img_wh[1] / self.meta["h"])

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.K) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            self.all_outfit_codes = []

            for t, frame in enumerate(self.meta['frames']):
                pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.FloatTensor(pose)
                outfit_code = frame['outfit_code']
                
                image_path = os.path.join(self.root_dir, frame["file_path"])
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = t * torch.ones(len(rays_o), 1)

                rays = torch.cat([rays_o, rays_d,
                    self.near * torch.ones_like(rays_o[:, :1]),  # Near bounds
                    self.far * torch.ones_like(rays_o[:, :1]), # Far bounds
                    rays_t ],
                    1)  # (H*W, 9)
                
                self.all_rays.append(rays)
                outfit_code_tensor = outfit_code * torch.ones(len(rays), dtype=torch.long)
                self.all_outfit_codes.append(outfit_code_tensor)

            # Concatenate all rays and RGB values
            self.all_rays = torch.cat(self.all_rays, 0)  # (N_images*H*W, 9)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (N_images*H*W, 3)
            self.all_outfit_codes = torch.cat(self.all_outfit_codes, 0)  # (N_images*H*W, 1)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx],
                      'outfit_code': self.all_outfit_codes[idx].squeeze(-1)
                    }

        else: # create data for each image separately
            
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            t = 0 # transient embedding index, 0 for val and test (no perturbation)

            img = Image.open(os.path.join(self.root_dir, frame["file_path"])).convert("RGB")
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # to tensor with shape (3, H, W))
            img = img.view(3, -1).permute(1, 0)  # Shape: (H*W, 3)
            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'rgbs': img,
                      'outfit_code': frame['outfit_code'] * torch.ones(len(rays), dtype=torch.long),
                      'c2w': c2w,
                      }

        return sample