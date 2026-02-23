import argparse
import os
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from safetensors.torch import load_model
from tqdm import tqdm

from powerpaint.datasets.fsc_147 import build_index_val, FSCDataset, BucketBatchSampler
from powerpaint.pipelines import StableDiffusionInpaintIndomainPipeline
from powerpaint.utils.utils import TokenizerWrapper, add_tokens, expand_unet_conv_in

from eval.pipe_counterfactual import infer_counterfactual_3



def parse_args():
    parser = argparse.ArgumentParser(description="Inference script using a trained PowerPaint checkpoint.")
    parser.add_argument("--config", type=str, default=None, help="Optional yaml config used during training.")
    parser.add_argument("--infer_txt", type=str, default="val_2.txt", help="Text file listing validation sample subdirectories.")
    parser.add_argument("--output_folder_name", type=str, default="inference", help="Where to write prediction results")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # load yaml config if provided (this will add attributes such as
    # base_model_path, task_prompt, train_root etc. to the args namespace)
    if args.config is not None:
        print(f"Loading config from {args.config}")
        conf = OmegaConf.load(args.config)
        for k, v in conf.items():
            setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the pipeline exactly as in training
    pipe = StableDiffusionInpaintIndomainPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained=args.base_model_path,
        subfolder="tokenizer",
        torch_dtype=torch.float32,
        local_files_only=True,
    )

    # make sure learned tokens are present (the config should contain
    # the same placeholder list used during training)
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
    )
    # the training script expanded conv_in, replicate the change here
    pipe.unet = expand_unet_conv_in(pipe.unet, extra_in_channels=1, init="mean_scaled")

    # load the checkpoint weights that were produced during training
    load_model(pipe.unet, os.path.join(args.checkpoint, "unet/diffusion_pytorch_model.safetensors"))
    load_model(pipe.text_encoder, os.path.join("/mnt/disk1/aiotlab/hachi/checkpoints/ppt-v1/text_encoder/text_encoder.safetensors"), strict=False)

    
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    # prepare validation dataloader
    by_bucket_sizes = {
        tuple(item["size"]): item["value"]
        for item in args.by_bucket_sizes
    }
    infer_items = build_index_val(args.infer_txt, args.density_root)
    if not hasattr(args, "task_prompt") or args.task_prompt is None:
        raise ValueError("task_prompt is required for dataset construction. Provide --config pointing to the training YAML.")
    infer_dataset = FSCDataset(infer_items, pipeline=pipe, task_prompt=args.task_prompt, train=False)
    infer_sampler = BucketBatchSampler(
        dataset=infer_dataset,
        by_bucket_sizes=by_bucket_sizes,
        shuffle=False,
        drop_last=False,
        bucket_sampling="proportional"
    )
    infer_dataloader = DataLoader(
        dataset=infer_dataset,
        batch_sampler=infer_sampler,
        num_workers=args.dataloader_num_workers,
    )
    
    out_dir = os.path.join(args.output_dir, args.output_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for bi, batch in tqdm(enumerate(infer_dataloader), total=len(infer_dataloader)):
            stats = infer_counterfactual_3(
                        pipe, batch,
                        out_dir=out_dir,
                        global_step=0,
                        accelerator=None,
                        seed=1234 + bi,
                        tradoff=1.0,
                        tradoff_nag=1.0,
                        save_k=4,
                    )
            

if __name__ == "__main__":
    main()
