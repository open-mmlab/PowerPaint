import torch
from powerpaint.pipelines.pipeline_indomain_ppt1 import StableDiffusionInpaintIndomainPipeline

device = "cuda"

# folder checkpoint đã save kiểu diffusers (vae/, unet/, text_encoder/, tokenizer/, scheduler/)
ckpt_dir = "/path/to/your/output_dir_or_checkpoint"

pipe = StableDiffusionInpaintIndomainPipeline.from_pretrained(
    ckpt_dir,
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
).to(device)

pipe.set_progress_bar_config(disable=True)

@torch.no_grad()
def infer_one_batch(pipe, batch, *, promptA, promptB, steps=30, seed=1234, tradoff=1.0, tradoff_nag=1.0):
    # batch tensors (theo doc preprocess: image [-1,1], mask [0,1]) :contentReference[oaicite:2]{index=2}
    image = batch["pixel_values"].to(device=device, dtype=torch.float32)   # [-1, 1]
    mask  = batch["mask"].to(device=device, dtype=torch.float32)           # [0, 1] (pipeline sẽ binarize) :contentReference[oaicite:3]{index=3}
    density = batch["density"].to(device=device, dtype=torch.float32)      # [0, 1] (kênh extra 10ch) :contentReference[oaicite:4]{index=4}

    # bucket size (nếu bạn có)
    if "bucket_hw" in batch:
        H, W = batch["bucket_hw"]
        if isinstance(H, torch.Tensor): H = int(H[0].item())
        if isinstance(W, torch.Tensor): W = int(W[0].item())
    else:
        # hoặc tự set theo data của bạn
        H, W = image.shape[-2], image.shape[-1]

    gen = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        promptA=promptA,
        promptB=promptB,
        image=image,
        mask=mask,
        density=density,
        height=H,
        width=W,
        strength=1.0,
        tradoff=tradoff,           # pipeline dùng để mix embed A/B :contentReference[oaicite:5]{index=5}
        tradoff_nag=tradoff_nag,   # mix negative A/B :contentReference[oaicite:6]{index=6}
        num_inference_steps=steps,
        guidance_scale=7.5,        # <=1.0 => không CFG (do_classifier_free_guidance=False) :contentReference[oaicite:7]{index=7}
        negative_promptA="",
        negative_promptB="",
        generator=gen,
        output_type="pt",          # nếu version diffusers bạn support; không thì dùng "np" hoặc "pil"
        return_dict=True,
    )

    # out.images thường là torch [B,3,H,W] trong [0,1] (vì postprocess) :contentReference[oaicite:8]{index=8}
    return out.images

# ví dụ:
# imgs = infer_one_batch(pipe, batch, promptA="...", promptB="...", steps=30, seed=1234, tradoff=0.8, tradoff_nag=0.8)
@torch.no_grad()
def log_validation(pipe, accelerator, task_prompt, batch, step, tradoff=1.0):
    pipe.set_progress_bar_config(disable=True)
    pipe.eval()

    # batch tensors (theo doc preprocess: image [-1,1], mask [0,1]) :contentReference[oaicite:2]{index=2}
    image = batch["pixel_values"].to(device=device, dtype=torch.float32)   # [-1, 1]
    mask  = batch["mask"].to(device=device, dtype=torch.float32)           # [0, 1] (pipeline sẽ binarize) :contentReference[oaicite:3]{index=3}
    density = batch["density"].to(device=device, dtype=torch.float32)      # [0, 1] (kênh extra 10ch) :contentReference[oaicite:4]{index=4}

    # bucket size (nếu bạn có)
    if "bucket_hw" in batch:
        H, W = batch["bucket_hw"][0]
        if isinstance(H, torch.Tensor): H = int(H[0].item())
        if isinstance(W, torch.Tensor): W = int(W[0].item())
    else:
        # hoặc tự set theo data của bạn
        H, W = image.shape[-2], image.shape[-1]

    bz = image.shape[0]
    promptA = []
    promptB = []
    for i in range(bz):
        prompt = batch["prompt"][i]
        promptA.append(f"{task_prompt.object_inpainting.placeholder_tokens} {prompt}")
        promptB.append(f"{task_prompt.object_inpainting.placeholder_tokens} {prompt}")
    
    with torch.autocast(accelerator.device.type):
        out = pipe(
            promptA=promptA,
            promptB=promptB,
            image=image,
            mask=mask,
            density=density,
            height=H,
            width=W,  
        )


    
    

