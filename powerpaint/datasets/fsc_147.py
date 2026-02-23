import os
import json
import random
import math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image


# -----------------------------
# Buckets + transforms (from our earlier logic)
# -----------------------------
BUCKETS = [(512, 512), (512, 768), (512, 1024)]  # (H, W)

def choose_bucket(h: int, w: int) -> Tuple[int, int]:
    ar = w / h
    best = None
    best_d = 1e9
    for Hb, Wb in BUCKETS:
        ar_b = Wb / Hb
        d = abs(math.log(ar / ar_b))
        if d < best_d:
            best_d = d
            best = (Hb, Wb)
    return best

def resize_cover(img: torch.Tensor, out_h: int, out_w: int, interp: InterpolationMode) -> torch.Tensor:
    # img: [C,H,W]
    _, h, w = img.shape
    scale = max(out_h / h, out_w / w)
    new_h = int(math.ceil(h * scale))
    new_w = int(math.ceil(w * scale))
    return TF.resize(img, [new_h, new_w], interpolation=interp, antialias=True)

def crop(img: torch.Tensor, top: int, left: int, out_h: int, out_w: int) -> torch.Tensor:
    return img[:, top:top + out_h, left:left + out_w]

def bucket_transform(
    image: torch.Tensor,   # [3,H,W] float in [0,1]
    mask: torch.Tensor,    # [1,H,W] 0/1
    density: torch.Tensor, # [1,H,W] float
    bucket_hw: Tuple[int, int],
    train: bool = True,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Hb, Wb = bucket_hw
    if rng is None:
        rng = random

    image_r = resize_cover(image, Hb, Wb, InterpolationMode.BICUBIC)
    density_r = resize_cover(density, Hb, Wb, InterpolationMode.BILINEAR)
    mask_r = resize_cover(mask, Hb, Wb, InterpolationMode.NEAREST)

    _, Hr, Wr = image_r.shape
    if train:
        top = rng.randint(0, Hr - Hb)
        left = rng.randint(0, Wr - Wb)
    else:
        top = (Hr - Hb) // 2
        left = (Wr - Wb) // 2

    image_c = crop(image_r, top, left, Hb, Wb)
    density_c = crop(density_r, top, left, Hb, Wb)
    mask_c = crop(mask_r, top, left, Hb, Wb)

    # safety: keep mask binary
    mask_c = (mask_c > 0.5).float()
    return image_c, mask_c, density_c

def preprocess_density(
    density: torch.Tensor,  # [1,H,W]
    c: float = 0.023632803931832314,  # p99.9 from your stats
    use_log: bool = False,
    k: float = 10.0
) -> torch.Tensor:
    d = density.clamp(min=0.0)
    d = (d.clamp(max=c) / c)  # [0,1]
    if use_log:
        d = torch.log1p(d * k) / math.log1p(k)
    return d

def bbox_to_mask_xyxy(
    bbox: List[float],
    h: int,
    w: int,
    expand_ratio: Optional[float] = 0.1,   # giãn theo %
    expand_pixels: Optional[int] = None    # hoặc giãn theo pixel
) -> torch.Tensor:
    """
    bbox: [x1, y1, x2, y2] in pixel coords (xyxy).
    expand_ratio: expand each side by ratio * bbox_size (default 10%)
    expand_pixels: expand each side by fixed pixels (override ratio if set)
    Returns mask [1,H,W] float {0,1}.
    """

    x1, y1, x2, y2 = bbox

    # --- compute expansion ---
    bw = x2 - x1
    bh = y2 - y1

    if expand_pixels is not None:
        dx = dy = expand_pixels
    else:
        dx = bw * expand_ratio
        dy = bh * expand_ratio

    # --- expand bbox ---
    x1 -= dx
    x2 += dx
    y1 -= dy
    y2 += dy

    # --- round ---
    x1 = int(math.floor(x1))
    y1 = int(math.floor(y1))
    x2 = int(math.ceil(x2))
    y2 = int(math.ceil(y2))

    # --- clamp ---
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    # --- create mask ---
    m = torch.zeros((1, h, w), dtype=torch.float32)
    if x2 > x1 and y2 > y1:
        m[:, y1:y2, x1:x2] = 1.0

    return m

# -----------------------------
# Index building from your folder structure
# -----------------------------
@dataclass
class SampleItem:
    sample_id: str
    img_path: str
    ann_path: str
    density_path: str
    orig_hw: Tuple[int, int]    # (H, W)
    bucket_hw: Tuple[int, int]  # (Hb, Wb)

def build_index(train_root: str, density_root: str, max_num: int = None) -> List[SampleItem]:
    """
    train_root:
      train/img1/img.png
      train/img1/annotation.json
    density_root:
      Density/img1.npy
    """
    items: List[SampleItem] = []
    subdirs = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])

    for i, sid in enumerate(subdirs):
        if max_num:
            if i > max_num:
                break
        img_path = os.path.join(train_root, sid, "ground_truth.jpg")
        ann_path = os.path.join(train_root, sid, "annotation.json")
        den_path = os.path.join(density_root, f"{sid.split('_')[0]}.npy")

        # read size cheaply
        with Image.open(img_path) as im:
            w, h = im.size

        bucket_hw = choose_bucket(h, w)
        items.append(SampleItem(
            sample_id=sid,
            img_path=img_path,
            ann_path=ann_path,
            density_path=den_path,
            orig_hw=(h, w),
            bucket_hw=bucket_hw,
        ))

    if len(items) == 0:
        raise RuntimeError("No valid samples found. Check paths and filenames.")
    return items

def build_index_val(val_txt: str, density_root: str, max_num: int = None) -> List[SampleItem]:
    items: List[SampleItem] = []

    subdirs = []
    with open(val_txt, 'r') as f:
        for line in f:
            subdirs.append(line.strip())

    for i, sid in enumerate(subdirs):
        if max_num:
            if i > max_num:
                break
        img_path = os.path.join(sid, "ground_truth.jpg")
        ann_path = os.path.join(sid, "annotation.json")
        den_path = os.path.join(density_root, f"{sid.split('/')[-1][:-3]}.npy")

        # read size cheaply
        with Image.open(img_path) as im:
            w, h = im.size

        bucket_hw = choose_bucket(h, w)
        items.append(SampleItem(
            sample_id=sid.split('/')[-1],
            img_path=img_path,
            ann_path=ann_path,
            density_path=den_path,
            orig_hw=(h, w),
            bucket_hw=bucket_hw,
        ))

    if len(items) == 0:
        raise RuntimeError("No valid samples found. Check paths and filenames.")
    return items


# -----------------------------
# Dataset
# -----------------------------
class FSCDataset(Dataset):
    """
    Diffusers-friendly training dataset for SD1.5 inpainting + extra density channel.
    Returns dict with:
      - pixel_values: float tensor [3,Hb,Wb] in [-1, 1]
      - mask: float tensor [1,Hb,Wb] in {0,1} (binary)
      - density: float tensor [1,Hb,Wb] in [0,1] after preprocessing
      - prompt: str
      - bucket_hw: (Hb, Wb)
      - id: optional identifier
    """

    def __init__(
        self,
        items: List[SampleItem],
        pipeline,
        task_prompt,
        *,
        density_clip_c: float = 0.023632803931832314,
        density_use_log: bool = False,
        density_log_k: float = 10.0,
        density_dropout_p: float = 0.0,  # set >0.0 if you want dataset-level dropout
        mask_threshold: int = 127,
        train: bool = True,
        seed: int = 0,
    ):
        self.pipeline = pipeline
        self.task_prompt = task_prompt

        self.items = items
        self.density_clip_c = float(density_clip_c)
        self.density_use_log = bool(density_use_log)
        self.density_log_k = float(density_log_k)
        self.density_dropout_p = float(density_dropout_p)

        self.mask_threshold = int(mask_threshold)
        self.train = bool(train)

        # For fast bucketing in sampler
        self.bucket_hw_list = [it.bucket_hw for it in items]

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        it = self.items[idx]

        # load image
        image = Image.open(it.img_path).convert("RGB")
        image = TF.to_tensor(image)  # [3,H,W] in [0,1]

        # load annotation
        with open(it.ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        prompt = ann.get("class_based_caption", "")
        ## bbox
        bboxes = ann.get("inpainted_bboxes", None)
        if bboxes is None:
            raise ValueError(f"Missing 'bbox' in {it.ann_path}")
        bbox = bboxes[0]

        H, W = image.shape[-2:]
        mask = bbox_to_mask_xyxy(bbox, H, W)  # [1,H,W]

        # load density
        d = np.load(it.density_path).astype(np.float32)
        if d.ndim == 3:
            if d.shape[0] == 1:
                d = d[0]
            elif d.shape[-1] == 1:
                d = d[..., 0]
            else:
                d = d[..., 0]
        density = torch.from_numpy(d)[None, ...]  # [1,H,W]

        # sanity alignment
        if density.shape[-2:] != (H, W):
            raise ValueError(f"Density shape mismatch for {it.sample_id}: density {tuple(density.shape)} vs image {(H,W)}")
        
        # preprocess density
        density = preprocess_density(
            density,
            c=self.density_clip_c,
            use_log=self.density_use_log,
            k=self.density_log_k,
        )

        # optional density dropout
        if self.train:
            if self.density_dropout_p > 0.0 and self._rng.random() < self.density_dropout_p:
                density = torch.zeros_like(density)

        # bucket resize+crop (aligned)
        image, mask, density = bucket_transform(
            image, mask, density,
            bucket_hw=it.bucket_hw,
            train=self.train,
            rng=self._rng,
        )

        # clamp numeric noise from interpolation – keeps pixel_values ∈ [-1,1]
        image = image.clamp(0.0, 1.0)
        
        # SD expects image normalized to [-1, 1] before VAE encode
        pixel_values = image * 2.0 - 1.0

        if self.train and (self._rng.random() < 0.0):
            prompt = ""

        promptA = self.task_prompt.object_inpainting.placeholder_tokens
        promptB = self.task_prompt.object_inpainting.placeholder_tokens
        promptA, promptB = f"{promptA} {prompt}", f"{promptB} {prompt}"
        prompt = self.pipeline.maybe_convert_prompt(prompt, self.pipeline.tokenizer)
        promptA = self.pipeline.maybe_convert_prompt(promptA, self.pipeline.tokenizer)
        promptB = self.pipeline.maybe_convert_prompt(promptB, self.pipeline.tokenizer)
        input_idsA, input_idsB, input_ids = self.pipeline.tokenizer(
            [promptA, promptB, prompt],
            max_length=self.pipeline.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        alpha = torch.tensor((1.0, 0.0))

        return {
            "pixel_values": pixel_values,   # [3,Hb,Wb] in [-1,1]
            "mask": mask,                   # [1,Hb,Wb] 0/1
            "density": density,             # [1,Hb,Wb] in [0,1]
            "prompt": prompt,
            "bucket_hw": it.bucket_hw,      # tuple(Hb,Wb)
            "id": it.sample_id,
            "input_idsA": input_idsA,
            "input_idsB": input_idsB,
            "input_ids": input_ids,
            "tradeoff": alpha,
        }

class BucketBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices, where each batch contains indices from the SAME bucket.
    Works with a dataset that returns sample["bucket_hw"] = (H, W).

    Two modes:
    - by_bucket_sizes=None: fixed batch_size for all buckets
    - by_bucket_sizes={(H,W): bs, ...}: variable batch size per bucket

    drop_last: drop incomplete batches inside each bucket.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        by_bucket_sizes: Optional[Dict[Tuple[int, int], int]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        max_samples_per_epoch: Optional[int] = None,
        bucket_sampling: str = "proportional", # "proportional" or "uniform"
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.by_bucket_sizes = by_bucket_sizes
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.max_samples_per_epoch = max_samples_per_epoch
        assert bucket_sampling in ["proportional", "uniform"]
        self.bucket_sampling = bucket_sampling

        # Precompute bucket -> list of indices
        self.bucket_to_indices = defaultdict(list)
        for i in range(len(dataset)):
            # dataset __getitem__ may be expensive, so we try:
            # 1) if dataset has precomputed bucket meta, use it
            # 2) else fall back to a light call to dataset.get_bucket_hw(i) if you implement it
            # 3) else call dataset[i] once (slower) to read bucket_hw
            if hasattr(dataset, "bucket_hw_list"):
                bhw = dataset.bucket_hw_list[i]
            elif hasattr(dataset, "get_bucket_hw"):
                bhw = dataset.get_bucket_hw(i)
            else:
                bhw = dataset[i]["bucket_hw"]
            self.bucket_to_indices[tuple(bhw)].append(i)

        self.buckets = sorted(list(self.bucket_to_indices.keys()))

    def _bucket_bs(self, bucket_hw: Tuple[int, int]) -> int:
        if self.by_bucket_sizes is not None:
            return int(self.by_bucket_sizes[bucket_hw])
        return int(self.batch_size)
    
    def __len__(self) -> int:
        # approximate length in batches for one epoch
        total_batches = 0
        for b in self.buckets:
            n = len(self.bucket_to_indices[b])
            bs = self._bucket_bs(b)
            if self.drop_last:
                total_batches += n // bs
            else:
                total_batches += math.ceil(n/bs)
        return total_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)

        # Copy & shuffle indices within each bucket
        bucket_lists = {b: list(idxs) for b, idxs in self.bucket_to_indices.items()}
        if self.shuffle:
            for b in self.buckets:
                rng.shuffle(bucket_lists[b])

        # Create per-bucket batch queues
        bucket_batches = defaultdict(list)
        for b in self.buckets:
            bs = self._bucket_bs(b)
            idxs = bucket_lists[b]
            for j in range(0, len(idxs), bs):
                batch = idxs[j:j+bs]
                if len(batch) < bs and self.drop_last:
                    continue
                bucket_batches[b].append(batch)

        # Decide order of emitting buckets
        # proportional: bucket appears proportionally to number of batches it has
        # uniform: alternate buckets more evenly (oversamples small buckets if max_samples_per_epoch is None)
        bucket_order = []
        if self.bucket_sampling == "proportional":
            for b in self.buckets:
                bucket_order += [b] * len(bucket_batches[b])
        else:  # uniform
            # round-robin over buckets until all exhausted
            active = [b for b in self.buckets if len(bucket_batches[b]) > 0]
            k = 0
            while active:
                b = active[k % len(active)]
                bucket_order.append(b)
                k += 1
                # We'll pop batches later; remove if empty
                if len(bucket_batches[b]) <= 1:
                    active = [x for x in active if x != b]

        if self.shuffle:
            rng.shuffle(bucket_order)

        emitted = 0
        # Pop one batch at a time from the chosen bucket
        for b in bucket_order:
            if not bucket_batches[b]:
                continue
            batch = bucket_batches[b].pop()
            yield batch
            emitted += len(batch)
            if self.max_samples_per_epoch is not None and emitted >= self.max_samples_per_epoch:
                break