import os
import pdb
import random
import time

import cv2
import numpy as np
import torch
from petrel_client.client import Client
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms

from .utils import INVALID_OPEN_FLAG, save_log


try:
    from petrel_client.client import Client
except ImportError:
    print("Failed to import petrel_client. Please install it if you are using petrel-oss.")


def augment_images(image, mask, resolution):
    mask[mask > 128] = 255
    mask[mask <= 128] = 0
    mask = Image.fromarray(mask.astype("uint8"))
    resize = transforms.Resize((resolution))
    image = resize(image)
    mask = resize(mask)

    crop = RandomCrop(resolution)
    image, mask = crop(image, mask)

    # 50% chance of applying horizontal flip
    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)

    # convert the image and mask to tensors
    toT = transforms.ToTensor()
    image = toT(image)
    mask = toT(mask)
    mask[mask != 0] = 1

    # normalize the image with mean and std
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    image = normalize(image)

    return image.unsqueeze(0), mask.unsqueeze(0)


def random_warponly(img, sigma=15, patch=40):
    # Get the image shape
    if np.max(img) > 128:
        img = img / 255
    h, w = img.shape[:2]

    # Generate random displacement vectors
    dx = np.random.normal(0, sigma, (int(w / patch), int(h / patch)))
    dy = np.random.normal(0, sigma, (int(w / patch), int(h / patch)))

    dx = cv2.resize(dx, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    # Add the displacements to an identity grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # Warp the image using the displacement map
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    warped += img
    warped[warped > 0.5] = 1.0
    warped[warped <= 0.5] = 0.0

    warped = warped * 255.0
    return warped


def get_min_bounding_box(mask, pp=5):
    H = np.shape(mask)[0]
    W = np.shape(mask)[1]
    nonzero_indices = np.nonzero(mask)
    if len(nonzero_indices) == 0:
        return mask
    min_row = max(np.min(nonzero_indices[0]) - pp, 0)
    max_row = min(np.max(nonzero_indices[0]) + pp, H)
    min_col = max(np.min(nonzero_indices[1]) - pp, 0)
    max_col = min(np.max(nonzero_indices[1]) + pp, W)
    bounding_box = np.zeros_like(mask)
    bounding_box[min_row : max_row + 1, min_col : max_col + 1] = 255
    return bounding_box


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = transforms.RandomCrop.get_params(image, output_size=(self.size, self.size))
        image = transforms.functional.crop(image, *crop_params)
        if target is not None:
            target = transforms.functional.crop(target, *crop_params)
        return image, target


class OpenImageBLIPaug_Dataset(IterableDataset):
    """Load data from OpenImages.
    PowerPaint mainly uses openimages with its mask as training data for:
        - text-based object inpainting w/ object segmentation masks,,
        - shape-guided object inpainting w/ object segmentation masks,
        - context-aware (i.e., text-free) image inpainting, but w/ random object masks from other instances.
    """

    def __init__(
        self,
        transforms,
        data_tokenizer,
        anno_root=None,
        image_root=None,
        mask_root=None,
        bufsize=None,
        clip_score_threshold=None,
        aesthetic_score_threshold=0.5,
        resolution=None,
        deterministic=False,
        use_petreloss=False,
        **kwargs,
    ):
        super().__init__()
        assert anno_root is not None, "Please provide the path to the annotation files."

        # for data loading
        self.client = Client(enable_multi_cluster=True, enable_mc=True)

        # loading prompts
        self.anno_list = []
        for i in range(16):
            with open(os.path.join(anno_root, f"prompt_anno_{i}.txt"), "r", encoding="utf-8") as f:
                data = f.read()
            f.close()
            data = data.split("\n")
            self.anno_list += data[:-1]
        random.shuffle(self.anno_list)

        # segmentation mask used for training
        self.mask_root = mask_root
        self.image_root = image_root

        # for data sample
        self.bufsize = bufsize
        self.resolution = resolution
        self.epoch = -1
        self.deterministic = deterministic
        self.tokenizer = data_tokenizer

        # for data filter
        self.aesthetic_score_threshold = aesthetic_score_threshold
        self.clip_score_threshold = clip_score_threshold
        self.transforms = transforms

    def tokenize_captions(self, prompt, is_train=True):
        captions = []
        caption = prompt
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(f"Caption column `{'prompt'}` should contain either strings or lists of strings.")

        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def _sample_data(self, data_info):
        # load mask
        mask = Image.open(data_info["mask"])

        # load images
        img_bytes = self.client.get(data_info["img_path"])
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        images = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        images = Image.fromarray(np.uint8(images))
        w, h = images.size
        # filter out low-resolution images
        if w < 512 or h < 512:
            return None

        output = {}
        mask = mask.resize((w, h), Image.LANCZOS)
        mask = np.array(mask).astype(np.float32)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # dilate masks
        S_mask = len(np.nonzero(mask)[0])
        if S_mask == 0:
            aug_mask = mask
            alpha = torch.tensor((1.0, 0.0)).unsqueeze(0)
        else:
            if data_info["bbox_flag"]:
                aug_mask = get_min_bounding_box(mask, pp=2)
                if random.random() > 0.5:
                    aug_mask = random_warponly(
                        aug_mask, sigma=20 / 200 * (S_mask ** (0.5)), patch=max(60 / 200 * (S_mask ** (0.5)), 4)
                    )
            else:
                aug_mask = mask
            S_aug = len(np.nonzero(aug_mask)[0])
            Srate = S_mask / S_aug
            Srate = min(max(Srate, 0), 1)
            if data_info["remove_flag"]:
                alpha = torch.tensor((1.0, 0.0)).unsqueeze(0)
            else:
                alpha = torch.tensor((Srate, 1 - Srate)).unsqueeze(0)

        output["pixel_values"], output["mask"] = augment_images(images, aug_mask, self.resolution)
        output["input_idsA"] = self.tokenize_captions(data_info["promptA"])
        output["input_idsB"] = self.tokenize_captions(data_info["promptB"])
        output["input_idsC"] = self.tokenize_captions(data_info["promptC"])
        output["input_ids"] = self.tokenize_captions(data_info["prompt"])
        output["tradeoff"] = alpha
        return output

    def sample_data(self):
        buffer = []
        for idx, anno_info in enumerate(self.anno_list):
            bbox_flag = 0
            remove_flag = 0
            anno_info = anno_info.split(",")

            if random.random() < 0.5:
                # using exact object segmentation mask
                if anno_info[3] == INVALID_OPEN_FLAG:
                    anno_info[3] = ""
                promptA = "P_shape"
                promptB = "P_shape"
                prompt = anno_info[3]

            else:
                remove_flag = 1
                bbox_flag = 1
                # using bounding box as training mask
                if anno_info[3] == INVALID_OPEN_FLAG:
                    anno_info[3] = ""
                promptA = "P_obj"
                promptB = "P_obj"
                prompt = anno_info[3]

            image_name, mask_name = anno_info[0], anno_info[2]
            image_name = image_name[1:] if image_name.startswith("/") else image_name
            mask_name = mask_name[1:] if mask_name.startswith("/") else mask_name
            image_name = os.path.join(self.image_root, image_name)
            mask_name = os.path.join(self.mask_root, mask_name)
            # 10% dropout for unconditional training
            if random.random() < 0.1:
                promptA = ""
                promptB = ""
                prompt = ""

            data_info = {
                "img_path": image_name,
                "mask": mask_name,
                "promptA": promptA,
                "promptB": promptB,
                "prompt": prompt,
                "promptC": "P_abc",
                "remove_flag": remove_flag,
                "bbox_flag": bbox_flag,
            }

            if self.bufsize is None:
                try:
                    data = self._sample_data(data_info)
                    if data is None:
                        continue
                    else:
                        yield data
                except Exception:
                    print(f"Error in {data_info}")
                    continue

            elif len(buffer) < self.bufsize:
                pdb.set_trace()
                buffer.append(data_info)

            else:
                select_idx = random.randint(0, self.bufsize - 1)
                save_log(f"[Data] Idx/SelectIdx: {idx}/{select_idx} " f"Name: {data_info}")

                selected_data = buffer[select_idx]
                pipe_start_it = time.time()
                try:
                    data = self._sample_data(data_info)
                    yield data
                except Exception:
                    print(f"Error in {selected_data}")
                    continue

                pipe_end_it = time.time()
                save_log(f"[Pipe] {pipe_end_it - pipe_start_it:.3f}s")
                buffer[select_idx] = data_info

        for data_info in buffer:
            try:
                yield self._sample_data(data_info)
            except Exception:
                print(f"Error in {data_info}")
                continue

    def __iter__(self):
        for data in self.sample_data():
            yield data

    def __len__(self):
        return 999_999_999
