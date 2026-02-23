import os
import torch
from powerpaint.utils.infer import sample_ddim_inpaint_density
from powerpaint.utils.vis import save_img, save_gray, save_grid

@torch.no_grad()
def infer_val_counterfactual(pipe, unet, text_encoder, vae, val_dataloader, args, accelerator,
                             global_step: int, weight_dtype, num_batches=2, steps=30, seed=1234):
    if not accelerator.is_main_process:
        return

    out_dir = os.path.join(args.output_dir, "val_infer", f"step-{global_step}")
    os.makedirs(out_dir, exist_ok=True)

    unet.eval(); text_encoder.eval();

    deltas_zero, deltas_shuf = [], []

    for bi, batch in enumerate(val_dataloader):
        if bi >= num_batches: break
        device = accelerator.device

        # chạy 3 variant với cùng seed -> so sánh “đúng” (counterfactual)
        out_real = sample_ddim_inpaint_density(
            pipe=pipe, unet=unet, text_encoder=text_encoder, vae=vae,
            batch=batch, weight_dtype=weight_dtype, num_inference_steps=steps,
            seed=seed+bi, density_variant="real"
        )
        out_zero = sample_ddim_inpaint_density(
            pipe=pipe, unet=unet, text_encoder=text_encoder, vae=vae,
            batch=batch, weight_dtype=weight_dtype, num_inference_steps=steps,
            seed=seed+bi, density_variant="zero"
        )
        out_shuf = sample_ddim_inpaint_density(
            pipe=pipe, unet=unet, text_encoder=text_encoder, vae=vae,
            batch=batch, weight_dtype=weight_dtype, num_inference_steps=steps,
            seed=seed+bi, density_variant="shuf"
        )

        # input + mask + density để compose
        pv = batch["pixel_values"].to(device=device, dtype=weight_dtype)
        mask = batch["mask"].to(device=device, dtype=weight_dtype)
        density = batch["density"].to(device=device, dtype=weight_dtype)

        inp = ((pv.clamp(-1,1) + 1) * 0.5).float()
        m = (mask > 0.5).float()
        m3 = m.repeat(1,3,1,1)

        comp_real = out_real*m3 + inp*(1-m3)
        comp_zero = out_zero*m3 + inp*(1-m3)
        comp_shuf = out_shuf*m3 + inp*(1-m3)

        denom = m3.sum(dim=(1,2,3)).clamp_min(1.0)
        d0 = (torch.abs(comp_real-comp_zero)*m3).sum(dim=(1,2,3))/denom
        ds = (torch.abs(comp_real-comp_shuf)*m3).sum(dim=(1,2,3))/denom
        deltas_zero += d0.detach().cpu().tolist()
        deltas_shuf += ds.detach().cpu().tolist()

        # save vài ảnh
        k = min(4, comp_real.shape[0])
        for i in range(k):
            sid = batch.get("id", None)
            if isinstance(sid, (list, tuple)):
                sid = sid[i]
            elif isinstance(sid, torch.Tensor):
                sid = str(sid[i].item())
            sid = sid or f"b{bi}_i{i}"

            sub = os.path.join(out_dir, sid)
            os.makedirs(sub, exist_ok=True)
            save_img(os.path.join(sub, "00_input.png"), inp[i].cpu())
            save_gray(os.path.join(sub, "01_mask.png"), m[i].cpu())
            save_gray(os.path.join(sub, "02_density.png"), density[i].cpu())
            save_img(os.path.join(sub, "10_comp_real.png"), comp_real[i].cpu())
            save_img(os.path.join(sub, "11_comp_zero.png"), comp_zero[i].cpu())
            save_img(os.path.join(sub, "12_comp_shuf.png"), comp_shuf[i].cpu())
            grid = torch.stack([inp[i].cpu(), comp_real[i].cpu(), comp_zero[i].cpu(), comp_shuf[i].cpu()], dim=0)
            save_grid(os.path.join(sub, "grid.png"), grid, nrow=4)

    if deltas_zero:
        dz = torch.tensor(deltas_zero)
        ds = torch.tensor(deltas_shuf)
        stats = {
            "val/delta_real_zero_mean": dz.mean().item(),
            "val/delta_real_shuf_mean": ds.mean().item(),
            "val/delta_real_zero_p50": dz.median().item(),
            "val/delta_real_shuf_p50": ds.median().item(),
        }
        accelerator.log(stats, step=global_step)
        print(f"[infer_val] {stats} -> {out_dir}")

    unet.train(); text_encoder.train(); 
