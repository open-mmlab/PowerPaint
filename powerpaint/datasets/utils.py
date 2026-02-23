from typing import List, Dict
import torch

def collate_train(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)  # [B,3,H,W]
    masks = torch.stack([x["mask"] for x in batch], dim=0)                 # [B,1,H,W]
    density = torch.stack([x["density"] for x in batch], dim=0)            # [B,1,H,W]

    prompts = [x.get("prompt", "") for x in batch]
    bucket_hw = batch[0]["bucket_hw"]  # same within batch

    out = {
        "pixel_values": pixel_values,
        "mask": masks,
        "density": density,
        "prompts": prompts,
        "bucket_hw": bucket_hw,
    }
    # optional ids
    if "id" in batch[0]:
        out["ids"] = [x["id"] for x in batch]
    return out