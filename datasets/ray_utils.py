import torch
import numpy as np
from kornia import create_meshgrid
from einops import rearrange


@torch.cuda.amp.autocast(dtype=torch.float32) # autocast: automatically cast inputs to float32
def get_ray_directions(H, W, K, device='cpu', random=False, return_uv=False, flatten=True):
    
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    
    Inputs:
        H, W: image height and width
        K: (3,3) camera intrinsic matrix
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """

    grid = create_meshgrid(H, W, False, device=device) # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    if random:
        directions = torch.stack([  (u - cx + torch.rand_like(u)) / fx,
                                    (v - cy + torch.rand_like(v)) / fy,
                                    torch.ones_like(u)],    dim=-1)
    else:
        directions = torch.stack([  (u - cx + 0.5) / fx,
                                    (v - cy + 0.5) / fy,
                                    torch.ones_like(u)],    dim=-1)
    if flatten:
        directions = directions.reshape(-1, 3) # or alternatively: rearrange(directions, 'h w c -> (h w) c')
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    
    return directions

@torch.cuda.amp.autocast(dtype=torch.float32)
def get_rays(directions, c2w):
    """
    Get ray origin and directions in world coordinate for all pixels in one image from the camera coordinate.

    Inputs:
        directions: (H*W, 3) or (N, 3) ray directions in camera coordinate
        c2w: (3, 4) or (N, 3, 4) transformation matrix from camera to world coordinate

    Outputs:
        rays_o: (H*W, 3) or (N,3) ray origins in world coordinate
        rays_d: (H*W, 3) or (N,3) ray directions in world coordinate
    """

    if c2w.ndim == 2:
        rays_d = directions @ c2w[:, :3].T # (H*W, 3)
    else:
        rays_d = rearrange(directions, 'n c -> n 1 c') @ rearrange(c2w[..., :3], 'n a b -> n b a')
        rays_d = rearrange(rays_d, 'n 1 c -> n c')

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[..., 3].expand_as(rays_d)  # (N, 3)

    return rays_o, rays_d


def axisangle_to_R(v):
    """
    Convert an axis-angle vector to rotation matrix
    from https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py#L47

    Inputs:
        v: (3) or (B, 3)
    
    Outputs:
        R: (3, 3) or (B, 3, 3)
    """
    v_ndim = v.ndim
    if v_ndim==1:
        v = rearrange(v, 'c -> 1 c')
    zero = torch.zeros_like(v[:, :1]) # (B, 1)    
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], 1) # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], 1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1) # (B, 3, 3)

    norm_v = rearrange(torch.norm(v, dim=1)+1e-7, 'b -> b 1 1')
    eye = torch.eye(3, device=v.device)
    R = eye + (torch.sin(norm_v) / norm_v) * skew_v + ((1 - torch.cos(norm_v)) / norm_v**2 ) * (skew_v @skew_v) # (B, 3, 3)
    if v_ndim==1:
        R = rearrange(R, '1 c d -> c d')
    return R

def normalize(v):
    """Normalize a vector"""
    return v / np.linalg.norm(v) 

def average_poses(poses, pts3d=None):
    pass

def center_poses(poses, pts3d=None):
    pass

def create_spheric_poses(radius, mean_h, n_poses=120):
    pass
