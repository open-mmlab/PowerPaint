import argparse
import gc
import logging
import math
import os
import shutil
import inspect
from typing import List, Tuple
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from safetensors.torch import load_model
from tqdm.auto import tqdm
from transformers import PretrainedConfig

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from powerpaint.datasets.fsc_147 import FSCDataset, BucketBatchSampler, build_index, build_index_val
from powerpaint.datasets.utils import collate_train
from powerpaint.models import UNet2DConditionModel
from powerpaint.pipelines import StableDiffusionInpaintIndomainPipeline
from powerpaint.utils.utils import TokenizerWrapper, add_tokens, expand_unet_conv_in

from eval.pipe_counterfactual import infer_counterfactual_3
# if is_wandb_available():
#     import wandb
#     from dotenv import load_dotenv
#     load_dotenv()
#     wandb.login()

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="yaml for configuration",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/ppt1_sd15",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        required=False,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warm-up period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warm-up in the lr scheduler.",  # noqa: F401
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # use omegaconf to manage configurations
    if args.config is not None:
        config = OmegaConf.load(args.config)
        for k, v in config.items():
            args.__dict__[k] = v
    return args

@torch.no_grad()
def vae_encode(vae, pixel_values: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents

@torch.no_grad()
def vae_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample #[-1, 1]
    return image

# def to_latent_mask(mask: torch.Tensor, lh: int, lw: int) -> torch.Tensor:
#     return F.interpolate(mask, size=(lh, lw), mode="nearest")

def to_latent_mask(
    mask: torch.Tensor,
    lh: int,
    lw: int,
    *,
    soft: bool = False,
    blur_kernel: int = 0,   # 0 = no blur, 3/5/7... = avg blur kernel size
    blur_iters: int = 1,    # số lần blur (1-3 thường đủ)
) -> torch.Tensor:
    """
    mask: [B,1,H,W] hoặc [B,H,W] hoặc [H,W]
    return: [B,1,lh,lw] float in [0,1]
      - soft=False  -> nearest, mask gần như nhị phân
      - soft=True   -> bilinear (+ optional blur) để làm mềm biên
    """
    # --- normalize shape to [B,1,H,W]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"mask must be 2D/3D/4D, got shape={tuple(mask.shape)}")

    mask = mask.float()

    # --- resize
    if soft:
        # bilinear tạo soft edges (giá trị 0..1)
        mask = F.interpolate(mask, size=(lh, lw), mode="bilinear", align_corners=False)
        mask = mask.clamp(0.0, 1.0)

        # --- optional blur boundary (giảm seam/halo ở rìa)
        if blur_kernel and blur_kernel > 1:
            if blur_kernel % 2 == 0:
                raise ValueError("blur_kernel should be odd (e.g., 3,5,7).")
            pad = blur_kernel // 2
            for _ in range(max(1, blur_iters)):
                mask = F.avg_pool2d(mask, kernel_size=blur_kernel, stride=1, padding=pad)
            mask = mask.clamp(0.0, 1.0)
    else:
        # nearest giữ mask nhị phân (phù hợp conditioning chuẩn inpainting)
        mask = F.interpolate(mask, size=(lh, lw), mode="nearest")

    return mask

def to_latent_density(density: torch.Tensor, lh: int, lw: int) -> torch.Tensor:
    return F.interpolate(density, size=(lh,lw), mode="bilinear", align_corners=False)

def set_trainable_params(unet, text_encoder, vae, train_mode: str):
    """
    Set requires_grad for modules according to train_mode.

    Recommended modes for your use case (SD1.5 inpaint + extra density channel):
      - "conv_in":          train only unet.conv_in
      - "conv_in+down0":    train unet.conv_in + unet.down_blocks.0
      - "conv_in+down01":   train unet.conv_in + unet.down_blocks.0 + unet.down_blocks.1
      - "unet":             train full UNet
      - "unet+te":          train UNet + text encoder
      - "full":             train UNet + text encoder + VAE

    By default (common practice): freeze VAE + text encoder unless train_mode includes them.

    Returns:
      A dict summary with counts (trainable/all) for each component.
    """
    # ---- reset all ----
    if vae is not None:
        vae.requires_grad_(False)
        vae.eval()
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    unet.requires_grad_(False)
    unet.train()  # we still want UNet in train mode if any params train

    def _enable_prefix(module, prefix: str):
        for name, p in module.named_parameters():
            if name.startswith(prefix):
                p.requires_grad = True

    def _enable_any(module, predicate):
        for name, p in module.named_parameters():
            if predicate(name):
                p.requires_grad = True

    # ---- parse mode ----
    mode = train_mode.strip().lower()

    # UNet selections
    if mode in ["conv_in", "convin"]:
        _enable_prefix(unet, "conv_in")

    elif mode in ["conv_in+down0", "convin+down0", "conv_in_down0"]:
        _enable_prefix(unet, "conv_in")
        _enable_prefix(unet, "down_blocks.0")

    elif mode in ["conv_in+down01", "convin+down01", "conv_in_down01"]:
        _enable_prefix(unet, "conv_in")
        _enable_prefix(unet, "down_blocks.0")
        _enable_prefix(unet, "down_blocks.1")

    elif mode in ["unet", "full_unet"]:
        unet.requires_grad_(True)

    elif mode in ["unet+te", "unet+text", "unet+text_encoder"]:
        unet.requires_grad_(True)
        if text_encoder is None:
            raise ValueError("text_encoder is None but train_mode requests it.")
        text_encoder.requires_grad_(True)
        text_encoder.train()

    elif mode in ["full", "all", "unet+te+vae"]:
        unet.requires_grad_(True)
        if text_encoder is None:
            raise ValueError("text_encoder is None but train_mode requests it.")
        if vae is None:
            raise ValueError("vae is None but train_mode requests it.")
        text_encoder.requires_grad_(True)
        vae.requires_grad_(True)
        text_encoder.train()
        vae.train()

    # Optional extra modes you may find useful:
    # - "attn_lora_like": train only attention blocks (without LoRA)
    elif mode in ["attn", "attention"]:
        # Train all attention (self+cross) weights inside UNet
        # This is heavier than LoRA but still less than full unet.
        _enable_any(unet, lambda n: (".attn" in n) or ("attentions" in n) or ("transformer_blocks" in n))

    elif mode in ["conv_in+attn", "convin+attn"]:
        _enable_prefix(unet, "conv_in")
        _enable_any(unet, lambda n: (".attn" in n) or ("attentions" in n) or ("transformer_blocks" in n))

    else:
        raise ValueError(
            f"Unknown train_mode='{train_mode}'. "
            f"Supported: conv_in, conv_in+down0, conv_in+down01, unet, unet+te, full, attn, conv_in+attn."
        )
    
    # If UNet has no trainable params after selection, keep it eval to save small overhead
    if not any(p.requires_grad for p in unet.parameters()):
        unet.eval()

    # ---- return a small summary (useful for logs) ----
    def _count_params(module):
        if module is None:
            return (0, 0)
        all_n = sum(p.numel() for p in module.parameters())
        train_n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return train_n, all_n

    unet_train, unet_all = _count_params(unet)
    te_train, te_all = _count_params(text_encoder)
    vae_train, vae_all = _count_params(vae)

    return {
        "unet_trainable": unet_train,
        "unet_all": unet_all,
        "text_encoder_trainable": te_train,
        "text_encoder_all": te_all,
        "vae_trainable": vae_train,
        "vae_all": vae_all,
        "mode": train_mode,
    }

def get_trainable_params(unet, text_encoder=None, vae=None) -> List[torch.nn.Parameter]:
    """
    Return a flat list of trainable parameters (requires_grad=True) across modules.
    Use this to build optimizer / gradient clipping.
    """
    params: List[torch.nn.Parameter] = []
    for module in (unet, text_encoder, vae):
        if module is None:
            continue
        for p in module.parameters():
            if p.requires_grad:
                params.append(p)

    # Safety: avoid empty optimizer
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found. Check set_trainable_params(train_mode=...).")
    return params

# def infer_before(pipe, args, accelerator):
#     pass

# @torch.no_grad()
# def inpaint_with_density(pipe, batch_val, accelerator, num_inference_steps: int = 30, guidance_scale: float = 7.5):
#     pixel_values = batch_val["pixel_values"]
#     mask = batch_val["mask"]
#     density = batch_val["density"]

def weighted_latent_mse(model_pred, target, mask_latent, masked_weight=5.0, known_weight=1.0, eps=1e-8):
    """
    model_pred, target: [B, C, H, W]  (C thường = 4 với SD1.5 latent)
    mask_latent:        [B, 1, H, W]  (1 = masked/hole, 0 = known)
    returns:
      loss_mean: scalar
      loss_per_sample: [B]
    """
    # đảm bảo float + đúng device
    mask = mask_latent.to(dtype=model_pred.dtype, device=model_pred.device)
    # weight map: [B,1,H,W]
    w = known_weight + (masked_weight - known_weight) * mask
    # expand sang channel: [B,C,H,W]
    w = w.expand(-1, model_pred.shape[1], -1, -1)

    mse = (model_pred - target) ** 2  # [B,C,H,W]

    # weighted mean per-sample (normalize theo tổng weight)
    num = (mse * w).sum(dim=(1, 2, 3))
    den = w.sum(dim=(1, 2, 3)).clamp_min(eps)
    loss_per_sample = num / den  # [B]

    return loss_per_sample.mean(), loss_per_sample

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

     # If passed along, set the training seed now.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # saving training configuration to output_dir
        to_save_config = OmegaConf.create(vars(args))
        OmegaConf.save(config=to_save_config, f=os.path.join(args.output_dir, "training_config.yaml"))

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # ==========================================
    # setting models: load scheduler, tokenizer and models.
    # ==========================================
    pipe = StableDiffusionInpaintIndomainPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=weight_dtype,
        local_files_only=True
    )
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained=args.base_model_path,
        subfolder='tokenizer',
        torch_dtype=weight_dtype,
        local_files_only=True
    )

    # add pretrained learned task tokens into the tokenizer
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
    )
    # load ppt1 checkpoint
    load_model(pipe.unet, os.path.join(args.ppt1_checkpoint, "unet/unet.safetensors"), strict=False)
    load_model(pipe.text_encoder, os.path.join(args.ppt1_checkpoint, "text_encoder/text_encoder.safetensors"),strict=False )

    # # Infer to check
    # if accelerator.is_main_process:
    #     infer_before(pipe, args, accelerator)

    # add the expanded channel
    pipe.unet = expand_unet_conv_in(pipe.unet, extra_in_channels=1, init="mean_scaled")

    vae, tokenizer, noise_scheduler = pipe.vae, pipe.tokenizer, pipe.scheduler
    text_encoder, unet = pipe.text_encoder.to(torch.float32), pipe.unet.to(torch.float32)

    set_trainable_params(unet, text_encoder, vae, args.train_mode)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    sub_dir = "unet" if isinstance(model, type(unwrap_model(unet))) else "text_encoder"
                    if sub_dir == "unet":
                        model.register_to_config(in_channels=10)
                        model.save_pretrained(os.path.join(output_dir, sub_dir))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()


        accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    optimizer_cls = torch.optim.AdamW
    parameters = get_trainable_params(unet, text_encoder, vae)
    optimizer = optimizer_cls(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Preparing datasets and dataloader for training
    items = build_index(args.train_root, args.density_root)
    train_dataset = FSCDataset(items, pipeline=pipe, task_prompt=args.task_prompt)

    by_bucket_sizes = {
        tuple(item["size"]): item["value"]
        for item in args.by_bucket_sizes
    }
    sampler = BucketBatchSampler(
        dataset=train_dataset,
        by_bucket_sizes=by_bucket_sizes,
        shuffle=True,
        drop_last=True,
        seed=42,
        bucket_sampling="proportional"
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        num_workers=args.dataloader_num_workers,
    )

    val_items = build_index_val('./val_2.txt', args.density_root, max_num=100)
    # val_items = build_index(args.val_root, args.density_root, max_num=100)
    val_dataset = FSCDataset(val_items, pipeline=pipe, task_prompt=args.task_prompt, train=False)
    val_sampler = BucketBatchSampler(
        dataset=val_dataset,
        by_bucket_sizes=by_bucket_sizes,
        shuffle=False,
        drop_last=False,
        bucket_sampling="proportional"
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.dataloader_num_workers,
    )
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs*num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        pop_list = []
        for k, v in tracker_config.items():
            if not isinstance(v, (int, float, str, bool, torch.Tensor)):
                pop_list.append(k)
                logger.info(f"Removed {k} (type:{type(v)}) from tracker_config")
        for k in pop_list:
            tracker_config.pop(k)

        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"***** Running training for {args.tracker_project_name} *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {int(args.max_train_steps)}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.args.max_train_steps
    progress_bar = tqdm(
        range(0, int(args.max_train_steps)),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    unet.train()
    text_encoder.train()

    # Check what we set grad True
    for name, p in unet.named_parameters():
        if p.requires_grad:
            logger.info(f'{name} {p.shape}')

    for _ in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae_encode(vae, batch["pixel_values"].to(weight_dtype))
                bsz, _, lh, lw  = latents.shape

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # in mask, 1 for masked region, 0 for known region
                mask_image = batch["pixel_values"] * (batch["mask"] < 0.5)
                # convert the hole value from 0 to -1 due to value range [-1, 1]
                mask_image = mask_image - batch["mask"]
                mask_image_latents = vae_encode(vae, mask_image.to(weight_dtype))

                # mask/density to latent resolution
                # mask cho model input (binary)
                mask = to_latent_mask(batch["mask"], lh, lw, soft=False)
                density_latent = to_latent_density(batch["density"], lh, lw)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                model_input = torch.cat([noisy_latents, mask, mask_image_latents, density_latent], dim=1)

                # Get the text embedding for conditioning unet, (bs, 77, 768)
                encoder_hidden_statesA = text_encoder(batch["input_idsA"], return_dict=False)[0]
                encoder_hidden_statesB = text_encoder(batch["input_idsB"], return_dict=False)[0]
                tradeoff = batch["tradeoff"].unsqueeze(-1)
                encoder_hidden_states = (
                    tradeoff[:, 0:1, :] * encoder_hidden_statesA + tradeoff[:, 1:, :] * encoder_hidden_statesB.detach()
                )
                encoder_hidden_states = encoder_hidden_states.to(accelerator.unwrap_model(unet).dtype)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(model_input, timesteps, encoder_hidden_states, return_dict=False)[0]

                # # mask cho loss weighting (soft optional)
                # mask_w_latent = to_latent_mask(
                #     batch["mask"], lh, lw,
                #     soft=True,        # bool
                #     blur_kernel=5,  # 0/3/5
                #     blur_iters=2,    # 1-3
                # )
                # # hyperparams (gợi ý): masked_weight=5~10, known_weight=1
                # masked_w = getattr(args, "masked_loss_weight", 5.0)
                # known_w  = getattr(args, "known_loss_weight", 1.0)

                # loss
                # ## Loss forcus on mask hole
                # if args.snr_gamma is None:
                #     loss, _ = weighted_latent_mse(
                #         model_pred.float(),
                #         target.float(),
                #         mask_latent=mask_w_latent,              # mask đã là latent-res [B,1,lh,lw]
                #         masked_weight=masked_w,
                #         known_weight=known_w,
                #     )
                # else:
                #     snr = compute_snr(timesteps)
                #     mse_loss_weights = (
                #         torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                #     )  # [B]

                #     _, loss_per_sample = weighted_latent_mse(
                #         model_pred.float(),
                #         target.float(),
                #         mask_latent=mask_w_latent,
                #         masked_weight=masked_w,
                #         known_weight=known_w,
                #     )
                #     loss = (loss_per_sample * mse_loss_weights).mean()

                ## Loss origin
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps


                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if (global_step % args.checkpointing_steps == 0):
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # infer val
                        out_root = os.path.join(args.output_dir, "val_infer_cf")
                        for bi, batch in enumerate(val_dataloader):
                            stats = infer_counterfactual_3(
                                pipe, batch,
                                out_dir=out_root,
                                global_step=global_step,
                                accelerator=accelerator,
                                seed=1234 + bi,
                                tradoff=1.0,
                                tradoff_nag=1.0,
                                save_k=4,
                            )
                        # infer_val_counterfactual(pipe, unet, text_encoder, vae, val_dataloader, args, accelerator,
                        #      global_step=global_step, weight_dtype=weight_dtype,
                        #      num_batches=2, steps=30, seed=1234)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    main()
