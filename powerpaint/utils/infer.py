import inspect
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

def vae_encode_with_generator(vae, x, generator):
    # match pipeline: sample(generator=...) * scaling_factor
    lat = vae.encode(x).latent_dist.sample(generator=generator)
    return lat * vae.config.scaling_factor

def vae_decode_scaled(vae, latents):
    # match pipeline: decode(latents / scaling_factor)
    x = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return x

@torch.no_grad()
def sample_ddim_inpaint_density(
    *,
    pipe, unet, text_encoder, vae,
    batch,
    weight_dtype,
    num_inference_steps=30,
    guidance_scale=0.0,   # bạn đang không dùng CFG
    seed=1234,
    density_variant="real",
    eta=0.0,
):
    device = next(unet.parameters()).device

    # --- scheduler: mimic pipeline style ---
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if getattr(pipe.scheduler.config, "prediction_type", None) is not None:
        scheduler.register_to_config(prediction_type=pipe.scheduler.config.prediction_type)
    scheduler.set_timesteps(num_inference_steps, device=device)

    # extra_step_kwargs giống pipeline.prepare_extra_step_kwargs :contentReference[oaicite:3]{index=3}
    extra_step_kwargs = {}
    step_params = set(inspect.signature(scheduler.step).parameters.keys())
    if "eta" in step_params:
        extra_step_kwargs["eta"] = eta

    # --- inputs ---
    pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32)  # keep float32 like pipeline preprocess
    mask = batch["mask"].to(device=device, dtype=torch.float32)
    density = batch["density"].to(device=device, dtype=torch.float32)
    bsz = pixel_values.shape[0]

    mask = (mask > 0.5).float()  # pipeline binarizes mask :contentReference[oaicite:4]{index=4}

    if density_variant == "zero":
        density = torch.zeros_like(density)
    elif density_variant == "shuf":
        perm = torch.randperm(bsz, device=device)
        density = density[perm]

    # --- encoder_hidden_states (giữ cách bạn đang làm) ---
    hsA = text_encoder(batch["input_idsA"].to(device), return_dict=False)[0]
    hsB = text_encoder(batch["input_idsB"].to(device), return_dict=False)[0]
    tradeoff = batch["tradeoff"].to(device).unsqueeze(-1)  # [B,2,1]
    encoder_hidden_states = (tradeoff[:,0:1,:]*hsA + tradeoff[:,1:,:]*hsB).to(dtype=unet.dtype)

    # --- determinism: use one generator for BOTH VAE sampling + noise ---
    gen = torch.Generator(device=device).manual_seed(seed)
    if "generator" in step_params:
        extra_step_kwargs["generator"] = gen

    # masked_image: match pipeline masked_image = image*(mask<0.5) :contentReference[oaicite:5]{index=5}
    masked_image = pixel_values * (mask < 0.5)

    # VAE encodes (match pipeline scaling + generator)
    masked_image_latents = vae_encode_with_generator(vae, masked_image.to(dtype=weight_dtype), gen)
    latents_ref = vae_encode_with_generator(vae, pixel_values.to(dtype=weight_dtype), gen)
    _, _, lh, lw = latents_ref.shape

    # mask_latent: pipeline uses interpolate(mask) without mode => nearest by default :contentReference[oaicite:6]{index=6}
    mask_latent = F.interpolate(mask, size=(lh, lw))
    mask_latent = mask_latent.to(device=device, dtype=weight_dtype)

    # density_latent: match pipeline bilinear + align_corners=False :contentReference[oaicite:7]{index=7}
    density_latent = F.interpolate(density, size=(lh, lw), mode="bilinear", align_corners=False)
    density_latent = density_latent.to(device=device, dtype=weight_dtype)

    # init noise latents
    latents = torch.randn((bsz, 4, lh, lw), generator=gen, device=device, dtype=weight_dtype)

    for t in scheduler.timesteps:
        # match pipeline: scale_model_input(latents) BEFORE concat :contentReference[oaicite:8]{index=8}
        latents_in = scheduler.scale_model_input(latents, t)

        model_in = torch.cat([latents_in, mask_latent, masked_image_latents, density_latent], dim=1)  # 10ch
        noise_pred = unet(model_in, t, encoder_hidden_states, return_dict=False)[0]

        # (CFG nếu bạn cần sau này thì implement giống pipeline; hiện guidance_scale=0 nên bỏ)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False, **extra_step_kwargs)[0]

    out = vae_decode_scaled(vae, latents)
    out = (out.clamp(-1, 1) + 1) * 0.5
    return out