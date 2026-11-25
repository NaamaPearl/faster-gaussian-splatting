"""Multi-view Faster-GS rasterizer: V cameras per call, gradients reduced over the views in-kernel.

mode="batched": one preprocess / depth sort / tile sort / blend for all views.
mode="loop":    one pipeline per view inside the call (same backward reduction).
"""
import torch
from torch.autograd.function import once_differentiable

from . import _C


class _RasterizeViews(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, scales, rotations, opacities, sh0, sh_rest, w2c, intrinsics, cam_position, bg, settings):
        active_sh, width, height, near, far, aa, batched = settings
        image, saved, n_buckets = _C.forward(
            means, scales, rotations, opacities, sh0, sh_rest, w2c, intrinsics, cam_position, bg,
            active_sh, width, height, near, far, aa, batched)
        ctx.settings = settings
        ctx.n_buckets = n_buckets
        ctx.save_for_backward(image, means, scales, rotations, opacities, sh_rest, w2c, intrinsics, cam_position, bg,
                              *saved)
        return image

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_image):
        image, means, scales, rotations, opacities, sh_rest, w2c, intrinsics, cam_position, bg, *saved = ctx.saved_tensors
        active_sh, width, height, _, _, aa, batched = ctx.settings
        grads = _C.backward(grad_image, image, means, scales, rotations, opacities, sh_rest, w2c, intrinsics,
                            cam_position, bg, saved, ctx.n_buckets, active_sh, width, height, aa, batched)
        return (*grads, None, None, None, None, None)


def rasterize_views(means, scales, rotations, opacities, sh0, sh_rest, w2c, K, width, height, bg=None,
                    near=0.01, far=1e4, active_sh_bases=None, proper_antialiasing=False, mode="batched"):
    """Renders [V, 3, H, W].

    means [N, 3], scales [N, 3] (log), rotations [N, 4] (wxyz, unnormalized), opacities [N, 1] (logit),
    sh0 [N, 1, 3], sh_rest [N, K, 3], w2c [V, 4, 4] world-to-camera, K [V, 3, 3] in pixels."""
    assert mode in ("batched", "loop")
    w2c = w2c.float().contiguous()
    intrinsics = torch.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], -1).float().contiguous()
    cam_position = torch.linalg.inv(w2c)[:, :3, 3].contiguous()
    if bg is None:
        bg = torch.zeros(3, device=means.device)
    if active_sh_bases is None:
        active_sh_bases = 1 + sh_rest.shape[1]
    settings = (int(active_sh_bases), int(width), int(height), float(near), float(far), bool(proper_antialiasing),
                mode == "batched")
    return _RasterizeViews.apply(means.contiguous(), scales.contiguous(), rotations.contiguous(),
                                 opacities.contiguous(), sh0.contiguous(), sh_rest.contiguous(),
                                 w2c, intrinsics, cam_position, bg.float().contiguous(), settings)
