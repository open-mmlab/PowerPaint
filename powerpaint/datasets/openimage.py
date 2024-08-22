import os
import random

import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from petrel_client.client import Client
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms


logger = get_logger(__name__)

INVALID_OPEN_FLAG = "a 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911"


try:
    from petrel_client.client import Client
except ImportError:
    logger.info("Failed to import petrel_client. Please install it if you are using petrel-oss.")


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = transforms.RandomCrop.get_params(image, output_size=(self.size, self.size))
        image = transforms.functional.crop(image, *crop_params)
        if target is not None:
            target = transforms.functional.crop(target, *crop_params)
        return image, target


def augment_images(image, mask, resolution):
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    mask = Image.fromarray(mask.astype("uint8")).convert("L")

    resize = transforms.Resize((resolution))
    image, mask = resize(image), resize(mask)
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

    return image, mask


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
        pipeline,
        task_prompt,
        desc_prefix=False,
        name=None,
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

        self.name = name
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
        self.pipeline = pipeline
        self.task_prompt = task_prompt
        self.desc_prefix = desc_prefix

        # for data filter
        self.aesthetic_score_threshold = aesthetic_score_threshold
        self.clip_score_threshold = clip_score_threshold
        self.transforms = transforms

    def _sample_data(self, data_info):
        output = {}

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

        # load mask
        mask = Image.open(data_info["mask"]).convert("L")
        mask = mask.resize((w, h), Image.NEAREST)  # (0,255)
        mask = np.array(mask).astype(np.float32)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        object_size = mask.sum() / 255.0
        # filter out images without object
        if object_size == 0:
            return None

        # dilate the mask
        else:
            # using bounding box (with micro-aug) for object inpainting
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            if data_info["task_type"] == "object_inpainting":
                aug_mask = get_min_bounding_box(mask, pp=2)
                if random.random() > 0.5:
                    aug_mask = random_warponly(
                        aug_mask,
                        sigma=20 / 200 * (object_size ** (0.5)),
                        patch=max(60 / 200 * (object_size ** (0.5)), 4),
                    )
                alpha = torch.tensor((1.0, 0.0))

            # using exact object segmentation mask for shape-guided inpainting
            elif data_info["task_type"] == "shape_inpainting":
                # improve original mask
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
                object_size = mask.sum() / 255.0

                # shape-guided dilation
                ksize = random.choice([ks for ks in range(3, 25) if ks % 2 == 1])
                iters = random.choice(range(0, 10))
                kernel = np.ones((ksize, ksize), np.uint8)
                aug_mask = cv2.dilate(mask, kernel, iters)
                _, aug_mask = cv2.threshold(aug_mask, 0, 255, cv2.THRESH_BINARY)

                mask_size = aug_mask.sum() / 255.0
                rate = object_size / mask_size
                rate = min(max(rate, 0), 1)
                alpha = torch.tensor((rate, 1 - rate))

            else:
                raise ValueError(f"Invalid task type: {data_info['task_type']}")

        output["pixel_values"], output["mask"] = augment_images(images, aug_mask, self.resolution)

        # filter data without meaningful masks (can be caused by randomcrop)
        if len(torch.unique(output["mask"])) == 1:
            return None

        output["tradeoff"] = alpha

        # IMPORTANT, remember to convert prompt for multi-vector embeddings
        promptA = self.pipeline.maybe_convert_prompt(data_info["promptA"], self.pipeline.tokenizer)
        promptB = self.pipeline.maybe_convert_prompt(data_info["promptB"], self.pipeline.tokenizer)
        prompt = self.pipeline.maybe_convert_prompt(data_info["prompt"], self.pipeline.tokenizer)

        output["input_idsA"], output["input_idsB"], output["input_ids"] = self.pipeline.tokenizer(
            [promptA, promptB, prompt],
            max_length=self.pipeline.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return output

    def sample_data(self):
        buffer = []
        for _, anno_info in enumerate(self.anno_list):
            anno_info = anno_info.split(",")
            if anno_info[3] == INVALID_OPEN_FLAG:
                continue

            prompt = anno_info[3]
            if random.random() < 0.5:
                # using bounding box as training mask for object inpainting
                # bbox-inpaint: obj + desc
                task_type = "object_inpainting"
                promptA = self.task_prompt.object_inpainting.placeholder_tokens
                promptB = self.task_prompt.object_inpainting.placeholder_tokens
            else:
                # using exact object segmentation mask for shape-guided inpainting
                task_type = "shape_inpainting"
                promptA = self.task_prompt.shape_inpainting.placeholder_tokens
                promptB = self.task_prompt.context_inpainting.placeholder_tokens

            # let see: NULL + obj or shape
            if random.random() < 0.3:
                prompt = ""

            if self.desc_prefix and prompt != "":  # for unet-based models
                promptA, promptB = f"{promptA} {prompt}", f"{promptB} {prompt}"

            image_name, mask_name = anno_info[0], anno_info[2]
            image_name = image_name[1:] if image_name.startswith("/") else image_name
            mask_name = mask_name[1:] if mask_name.startswith("/") else mask_name
            image_name = os.path.join(self.image_root, image_name)
            mask_name = os.path.join(self.mask_root, mask_name)

            # 10% dropout for unconditional training
            if random.random() < 0.1:
                promptA = promptB = prompt = ""

            data_info = {
                "img_path": image_name,
                "mask": mask_name,
                "promptA": promptA,
                "promptB": promptB,
                "prompt": prompt,
                "task_type": task_type,
            }

            if self.bufsize is None:
                try:
                    data = self._sample_data(data_info)
                    if data is None:
                        continue
                    else:
                        yield data
                except Exception:
                    logger.info(f"Error in {data_info}")
                    continue

            elif len(buffer) < self.bufsize:
                buffer.append(data_info)

            else:
                select_idx = random.randint(0, self.bufsize - 1)

                selected_data = buffer[select_idx]
                try:
                    data = self._sample_data(selected_data)
                    yield data
                except Exception:
                    logger.info(f"Error in {selected_data}")
                    continue

                buffer[select_idx] = data_info

        for data_info in buffer:
            try:
                yield self._sample_data(data_info)
            except Exception:
                logger.info(f"Error in {data_info}")
                continue

    def __iter__(self):
        for data in self.sample_data():
            yield data

    def __len__(self):
        return 999_999_999

    def __repr__(self):
        return f"OpenImageBLIPaug_Dataset(name={self.name}, resolution={self.resolution})"
