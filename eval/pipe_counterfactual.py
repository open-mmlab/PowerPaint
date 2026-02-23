import os
import math
import numpy as np
import torch
from PIL import Image

def _to_uint8_img(x_chw_01: torch.Tensor) -> np.ndarray:
    """x: [C,H,W] in [0,1] -> uint8 [H,W,C]"""
    x = x_chw_01.detach().clamp(0, 1).cpu()
    x = (x * 255.0 + 0.5).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    return x

def save_img(path: str, x_chw_01: torch.Tensor):
    arr = _to_uint8_img(x_chw_01)
    Image.fromarray(arr).save(path)

def save_gray(path: str, x_1hw_01: torch.Tensor):
    """x: [1,H,W] or [H,W] in [0,1] -> grayscale PNG"""
    x = x_1hw_01.detach().clamp(0, 1).cpu()
    if x.ndim == 3:
        x = x[0]
    arr = (x * 255.0 + 0.5).to(torch.uint8).numpy()
    Image.fromarray(arr, mode="L").save(path)

def save_grid(path: str, imgs_nchw_01: torch.Tensor, nrow: int = 4, pad: int = 2):
    """
    imgs: [N,3,H,W] in [0,1]
    grid = concat theo hàng, có padding trắng
    """
    imgs = imgs_nchw_01.detach().clamp(0, 1).cpu()
    N, C, H, W = imgs.shape
    nrow = max(1, nrow)
    ncol = min(nrow, N)
    nrows = math.ceil(N / nrow)

    grid_h = nrows * H + (nrows - 1) * pad
    grid_w = ncol * W + (ncol - 1) * pad
    grid = torch.ones((C, grid_h, grid_w), dtype=imgs.dtype)  # nền trắng

    for idx in range(N):
        r = idx // nrow
        c = idx % nrow
        top = r * (H + pad)
        left = c * (W + pad)
        grid[:, top:top+H, left:left+W] = imgs[idx]

    save_img(path, grid)

# @torch.no_grad()
# def infer_counterfactual_3(
#     pipe,
#     batch,
#     *,
#     out_dir: str,
#     global_step: int = 0,
#     accelerator=None,              # optional
#     seed: int = 1234,
#     tradoff: float = 1.0,
#     tradoff_nag: float = 1.0,
#     save_k: int = 4,               # lưu tối đa k ảnh trong batch
# ):
#     # chỉ main process mới save/log
#     if accelerator is not None and (not accelerator.is_main_process):
#         return None

#     pipe.safety_checker = None
#     device = next(pipe.unet.parameters()).device

#     # lấy H,W theo bucket nếu có
#     if "bucket_hw" in batch:
#         H, W = batch["bucket_hw"]
#         if isinstance(H, torch.Tensor): H = int(H[0].item())
#         if isinstance(W, torch.Tensor): W = int(W[0].item())
#     else:
#         H, W = batch["pixel_values"].shape[-2], batch["pixel_values"].shape[-1]
#     dtype = pipe.unet.dtype
#     image = batch["pixel_values"].to(device=device, dtype=dtype)  # [-1,1]
#     mask  = batch["mask"].to(device=device, dtype=dtype)          # [0,1]
#     density = batch["density"].to(device=device, dtype=dtype)     # [0,1]
#     bsz = image.shape[0]

#     promptA = []
#     promptB = []
#     for i in range(bsz):
#         prompt = batch["prompt"][i]
#         promptA.append(f"P_obj {prompt}")
#         promptB.append(f"P_obj {prompt}")

#     # để compose/log: input về [0,1]
#     inp = ((image.clamp(-1, 1) + 1) * 0.5).float()
#     m = (mask > 0.5).float()
#     m3 = m.repeat(1, 3, 1, 1)

#     def run_with_density(density_in: torch.Tensor):
#         with torch.autocast(accelerator.device.type):
#             out = pipe(
#                 promptA=promptA,
#                 promptB=promptB,
#                 image=image,
#                 mask=mask,
#                 density=density_in,
#                 height=H,
#                 width=W,
#                 tradoff=tradoff,
#                 tradoff_nag=tradoff_nag,
#                 output_type="pt",
#                 return_dict=True,
#             ).images
#             return out  # [B,3,H,W] in [0,1]

#     # 3 biến thể
#     out_real = run_with_density(density)

#     out_zero = run_with_density(torch.zeros_like(density))

#     perm = torch.randperm(bsz, device=device)
#     out_shuf = run_with_density(density[perm])

#     # compose vùng ngoài mask lấy input gốc
#     comp_real = out_real * m3 + inp * (1 - m3)
#     comp_zero = out_zero * m3 + inp * (1 - m3)
#     comp_shuf = out_shuf * m3 + inp * (1 - m3)

#     # delta trong vùng mask
#     denom = m3.sum(dim=(1, 2, 3)).clamp_min(1.0)
#     d0 = (torch.abs(comp_real - comp_zero) * m3).sum(dim=(1, 2, 3)) / denom
#     ds = (torch.abs(comp_real - comp_shuf) * m3).sum(dim=(1, 2, 3)) / denom

#     stats = {
#         "val/delta_real_zero_mean": d0.mean().item(),
#         "val/delta_real_shuf_mean": ds.mean().item(),
#         "val/delta_real_zero_p50": d0.median().item(),
#         "val/delta_real_shuf_p50": ds.median().item(),
#     }

#     # log
#     if accelerator is not None and hasattr(accelerator, "log"):
#         accelerator.log(stats, step=global_step)
#     print(f"[infer_cf] step={global_step} stats={stats}")

#     # save ảnh
#     step_dir = os.path.join(out_dir, f"step-{global_step}")
#     os.makedirs(step_dir, exist_ok=True)

#     k = min(save_k, bsz)
#     for i in range(k):
#         sid = batch.get("id", None)
#         if isinstance(sid, (list, tuple)):
#             sid = sid[i]
#         elif isinstance(sid, torch.Tensor):
#             sid = str(sid[i].item())
#         sid = sid or f"i{i}"

#         sub = os.path.join(step_dir, sid)
#         os.makedirs(sub, exist_ok=True)

#         save_img(os.path.join(sub, "00_input.png"), inp[i].cpu())
#         save_gray(os.path.join(sub, "01_mask.png"), m[i].cpu())
#         save_gray(os.path.join(sub, "02_density_real.png"), density[i].cpu())

#         # cũng lưu density_shuf/zero để debug
#         save_gray(os.path.join(sub, "02_density_zero.png"), torch.zeros_like(density[i]).cpu())
#         save_gray(os.path.join(sub, "02_density_shuf.png"), density[perm][i].cpu())

#         save_img(os.path.join(sub, "10_comp_real.png"), comp_real[i].cpu())
#         save_img(os.path.join(sub, "11_comp_zero.png"), comp_zero[i].cpu())
#         save_img(os.path.join(sub, "12_comp_shuf.png"), comp_shuf[i].cpu())

#         grid = torch.stack(
#             [inp[i].cpu(), comp_real[i].cpu(), comp_zero[i].cpu(), comp_shuf[i].cpu()],
#             dim=0
#         )
#         save_grid(os.path.join(sub, "grid.png"), grid, nrow=4, pad=2)

#     return stats



# -----------------------------
# Helpers: mask -> bbox, draw bbox on image tensor
# -----------------------------
def mask_to_bbox(mask_1hw: torch.Tensor, thresh: float = 0.5, pad: int = 2):
    """
    mask_1hw: [1,H,W] hoặc [H,W], float/bool
    return (x0, y0, x1, y1) theo pixel index (xyxy), hoặc None nếu mask rỗng
    """
    m = mask_1hw
    if m.ndim == 3:
        m = m[0]
    m = (m > thresh)

    ys, xs = torch.where(m)
    if ys.numel() == 0:
        return None

    y0 = int(ys.min().item())
    y1 = int(ys.max().item())
    x0 = int(xs.min().item())
    x1 = int(xs.max().item())

    H, W = m.shape[-2], m.shape[-1]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return (x0, y0, x1, y1)


def draw_bbox(img_3hw: torch.Tensor, bbox, color=(1.0, 0.0, 0.0), thickness: int = 3):
    """
    img_3hw: [3,H,W] float trong [0,1]
    bbox: (x0,y0,x1,y1) hoặc None
    """
    if bbox is None:
        return img_3hw
    x0, y0, x1, y1 = bbox
    img = img_3hw.clone()

    c = torch.tensor(color, dtype=img.dtype, device=img.device).view(3, 1)  # [3,1]

    H, W = img.shape[-2], img.shape[-1]
    x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W - 1, x1))
    y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H - 1, y1))
    if x1 < x0 or y1 < y0:
        return img

    for t in range(thickness):
        yt0 = max(0, y0 - t)
        yt1 = min(H - 1, y1 + t)
        xt0 = max(0, x0 - t)
        xt1 = min(W - 1, x1 + t)

        # top & bottom
        img[:, yt0, xt0:xt1 + 1] = c
        img[:, yt1, xt0:xt1 + 1] = c
        # left & right
        img[:, yt0:yt1 + 1, xt0] = c
        img[:, yt0:yt1 + 1, xt1] = c

    return img


@torch.no_grad()
def infer_counterfactual_3(
    pipe,
    batch,
    *,
    out_dir: str,
    global_step: int = 0,
    accelerator=None,              # optional
    seed: int = 1234,
    tradoff: float = 1.0,
    tradoff_nag: float = 1.0,
    save_k: int = 4,               # lưu tối đa k ảnh trong batch
    bbox_pad: int = 2,
    bbox_thickness: int = 3,
):
    # chỉ main process mới save/log
    if accelerator is not None and (not accelerator.is_main_process):
        return None
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    device = next(pipe.unet.parameters()).device

    # lấy H,W theo bucket nếu có
    if "bucket_hw" in batch:
        H, W = batch["bucket_hw"]
        if isinstance(H, torch.Tensor): H = int(H[0].item())
        if isinstance(W, torch.Tensor): W = int(W[0].item())
    else:
        H, W = batch["pixel_values"].shape[-2], batch["pixel_values"].shape[-1]

    dtype = pipe.unet.dtype
    image   = batch["pixel_values"].to(device=device, dtype=dtype)  # [-1,1]
    mask    = batch["mask"].to(device=device, dtype=dtype)          # [0,1]
    density = batch["density"].to(device=device, dtype=dtype)       # [0,1]
    bsz = image.shape[0]

    promptA, promptB = [], []
    for i in range(bsz):
        prompt = batch["prompt"][i]
        promptA.append(f"P_obj {prompt}")
        promptB.append(f"P_obj {prompt}")

    # để compose/log: input về [0,1]
    inp = ((image.clamp(-1, 1) + 1) * 0.5).float()
    m = (mask > 0.5).float()
    m3 = m.repeat(1, 3, 1, 1)

    # --- chạy pipe (robust autocast ngay cả khi accelerator=None) ---
    autocast_device = (accelerator.device.type if accelerator is not None else device.type)
    gen = torch.Generator(device=device).manual_seed(seed)

    def run_with_density(density_in: torch.Tensor):
        with torch.autocast(autocast_device):
            out = pipe(
                promptA=promptA,
                promptB=promptB,
                image=image,
                mask=mask,
                density=density_in,
                height=H,
                width=W,
                tradoff=tradoff,
                tradoff_nag=tradoff_nag,
                output_type="pt",
                return_dict=True,
                generator=gen,  # giữ tính "counterfactual" theo seed
            ).images
            return out  # [B,3,H,W] in [0,1]

    # 1 biến thể
    out_real = run_with_density(density)
    out_zero = run_with_density(torch.zeros_like(density))

    # compose vùng ngoài mask lấy input gốc
    comp_real = out_real * m3 + inp * (1 - m3)
    comp_zero = out_zero * m3 + inp * (1 - m3)

    # delta trong vùng mask
    denom = m3.sum(dim=(1, 2, 3)).clamp_min(1.0)
    d0 = (torch.abs(comp_real - comp_zero) * m3).sum(dim=(1, 2, 3)) / denom

    stats = {
        "val/delta_real_zero_mean": d0.mean().item(),
        "val/delta_real_zero_p50": d0.median().item(),
    }

    # log
    if accelerator is not None and hasattr(accelerator, "log"):
        accelerator.log(stats, step=global_step)
    print(f"[infer_cf] step={global_step} stats={stats}")

    # save ảnh
    step_dir = os.path.join(out_dir, f"step-{global_step}")
    os.makedirs(step_dir, exist_ok=True)

    k = min(save_k, bsz)
    for i in range(k):
        sid = batch.get("id", None)
        if isinstance(sid, (list, tuple)):
            sid = sid[i]
        elif isinstance(sid, torch.Tensor):
            sid = str(sid[i].item())
        sid = sid or f"i{i}"

        sub = os.path.join(step_dir, sid)
        os.makedirs(sub, exist_ok=True)

        # --- compute bbox từ mask (CPU) ---
        mi = m[i].detach().cpu()  # [1,H,W]
        bbox = mask_to_bbox(mi, thresh=0.5, pad=bbox_pad)

        # --- tensors CPU để save + vẽ bbox ---
        inp_i  = inp[i].detach().cpu()
        cr_i   = comp_real[i].detach().cpu()
        cz_i   = comp_zero[i].detach().cpu()

        inp_bbox = draw_bbox(inp_i, bbox, thickness=bbox_thickness)
        cr_bbox  = draw_bbox(cr_i,  bbox, thickness=bbox_thickness)
        cz_bbox  = draw_bbox(cz_i,  bbox, thickness=bbox_thickness)

        # # lưu gốc
        # save_img(os.path.join(sub, "00_input.png"), inp_i)
        # save_gray(os.path.join(sub, "01_mask.png"), mi)
        save_gray(os.path.join(sub, "02_density_real.png"), density[i].detach().cpu())

        # cũng lưu density_shuf/zero để debug
        save_gray(os.path.join(sub, "02_density_zero.png"), torch.zeros_like(density[i]).detach().cpu())

        # save_img(os.path.join(sub, "10_comp_real.png"), cr_i)
        # save_img(os.path.join(sub, "11_comp_zero.png"), cz_i)

        # grid = torch.stack([inp_i, cr_i, cz_i], dim=0)
        # save_grid(os.path.join(sub, "grid.png"), grid, nrow=3, pad=2)

        # # lưu phiên bản có bbox
        # save_img(os.path.join(sub, "00_input_bbox.png"), inp_bbox)
        # save_img(os.path.join(sub, "10_comp_real_bbox.png"), cr_bbox)
        # save_img(os.path.join(sub, "11_comp_zero_bbox.png"), cz_bbox)

        grid_bbox = torch.stack([inp_bbox, cr_bbox, cz_bbox], dim=0)
        save_grid(os.path.join(sub, "grid_bbox.png"), grid_bbox, nrow=3, pad=2)

    return stats