import torch
import vren
from torch.cuda.amp import custom_fwd, custom_bwd
from torch_scatter import segment_csr
from einops import rearrange




class RayAABBIntersector(torch.autograd.Function):
    """
    Computes the intersection of rays with an axis-aligned bounding box (AABB).
    
    Inputs:
        rays_o: ray origins (N_rays, 3)
        rays_d: ray directions (N_rays, 3)
        centers: voxel centers (N_voxels, 3)
        half_sizes: voxel half sizes (N_voxels, 3)
        max_hits: maximum number of intersected voxels to keep for one ray 
                  (for a cubic scene, this is at most 3*N_voxels^(1/3)-2)

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits_, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit)
    """

    @staticmethod # static method: can be called without instantiating the class first
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return vren.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)
    
class RayMarcher(torch.autograd.Function):
    """
    March the rays to get sample point positions and directions.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) normalized ray directions
        hits_t: (N_rays, 2) near and far bounds from aabb intersection (aabb: axis-aligned bounding box)
        density_bitfield: (C*G**3//8) bitfield of density grids 
        cascades: number of cascades (int)
        scale: scene scale (float)
        exp_step_factor: the exponential factor to scale the step size
        grid_size: (int)
        max_samples: (int)

    Outputs: 
        rays_a: (N_rays, 3) each row corresponds to one ray and has three attributes: 
                    ray_idx     is the index of the ray in the input tensor (rays_o and rays_d) 
                    start_idx   is the index of the first sample of the ray in the xyzs, dirs, deltas, and ts tensors
                    N_samples   is the number of samples taken along the ray
        xyzs: (N_samples, 3) sample positions
        dirs: (N_samples, 3) sample view directions
        deltas: (N_samples) sample step sizes dt for integration
        ts: (N_samples) sample ts. ts is the distance from the ray origin to the sample position
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, hits_t, density_bitfield, cascades, scale, exp_step_factor, grid_size, max_samples):
        # noise to perturb the first sample of each ray
        noise = torch.rand_like(rays_o[:, 0]) 

        rays_a, xyzs, dirs, deltas, ts, counter = vren.raymarching_train(rays_o, rays_d, hits_t, 
                                                                         density_bitfield, cascades, 
                                                                         scale, exp_step_factor,
                                                                         noise, grid_size, max_samples)
        
        total_samples = counter[0]  # total number of samples for all rays
        # remove redundant output
        xyzs = xyzs[:total_samples]
        dirs = dirs[:total_samples]
        deltas = deltas[:total_samples]
        ts = ts[:total_samples]

        ctx.save_for_backward(rays_a, ts)

        return rays_a, xyzs, dirs, deltas, ts, total_samples
    
    @staticmethod
    @custom_bwd
    def backward(ctx, dL_drays_a, dL_dxyzs, dL_ddirs, dL_ddeltas, dL_dts, dL_dtotal_samples):
        rays_a, ts = ctx.saved_tensors
        segments = torch.cat([rays_a[:, 1], rays_a[-1:, 1]+rays_a[-1:, 2]]) 
        dL_drays_o = segment_csr(dL_dxyzs, segments)
        dL_drays_d = segment_csr(dL_dxyzs*rearrange(ts, 'n -> n 1') + dL_ddirs, segments)
        return dL_drays_o, dL_drays_d, None, None, None, None, None, None, None
    

class VolumeRenderer(torch.autograd.Function):
    """
    Volume rendering with different number of samples per ray.
    Only used in training.
    
    Inputs:
        sigmas: (N)
        rgbs: (N, 3)
        deltas: (N_samples) deltas for integration. Number of samples per ray is N and is same for each ray
        ts: (N_samples) distances from ray origins to sample positions
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
        T_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        total_samples: int, total effective samples
        opacity: (N_rays)
        depth: (N_rays)
        rgb: (N_rays, 3)
        ws: (N) sample point weights
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, deltas, ts, rays_a, T_threshold):
        total_samples, opacity, depth, rgb, ws = vren.composite_train_fw(sigmas, rgbs, deltas, ts, rays_a, T_threshold)
        ctx.save_for_backward(sigmas, rgbs, deltas, ts, rays_a, opacity, depth, rgb, ws)
        ctx.T_threshold = T_threshold
        return total_samples.sum(), opacity, depth, rgb, ws
    
    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dtotal_samples, dL_dopacity, dL_ddepth, dL_drgb, dL_dws):
        sigmas, rgbs, deltas, ts, rays_a, opacity, depth, rgb, ws = ctx.saved_tensors
        dL_dsigmas, dL_drgbs = vren.composite_train_bw(dL_dopacity, dL_ddepth, dL_drgb, dL_dws, sigmas, rgbs, ws, deltas, ts, rays_a, opacity, depth, rgb, ctx.T_threshold)
        return dL_dsigmas, dL_drgbs, None, None, None, None
    