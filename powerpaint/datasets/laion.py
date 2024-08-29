import json
import os
import os.path as osp
import random
import time

import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms
from webdataset import utils


logger = get_logger(__name__)

try:
    from petrel_client.client import Client
except ImportError:
    logger.info("Failed to import petrel_client. Please install it if you are using petrel-oss.")


class LaionIterJsonDataset(IterableDataset):
    """Load data from Laion.
    PowerPaint mainly uses laion as prompt-free training data for:
        - text-to-image generation
        - context-aware (i.e., text-free) image inpainting
        - image outpainting
    """

    def __init__(
        self,
        transforms,
        pipeline,
        task_prompt,
        desc_prefix=False,
        name=None,
        anno_root=None,
        random_mask_root=None,
        bufsize=None,
        clip_score_threshold=None,
        aesthetic_score_threshold=0.5,
        resolution=None,
        deterministic=False,
        client_prefix="",
        **kwargs,
    ):
        super().__init__()
        assert anno_root is not None, "Please provide the path to the annotation files."
        self.name = name

        # for data loading
        self.client_prefix = client_prefix
        self.client = Client(enable_multi_cluster=True, enable_mc=True)
        self.anno_list = []
        for anno in self.client.list(anno_root):
            if not anno.endswith(".jsonl"):
                continue
            self.anno_list.append(os.path.join(anno_root, anno))

        # random mask used for training
        random_mask_list = os.listdir(random_mask_root)
        self.random_mask_list = [os.path.join(random_mask_root, m) for m in random_mask_list]

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

    def _sample_anno(self):
        """Modified from https://github.com/webdataset/webdataset/blob/039d7431
        9ae55e5696dcef89829be9671802cf70/webdataset/shardlists.py#L281  #
        noqa."""
        self.epoch += 1
        if self.deterministic:
            seed = utils.make_seed(utils.pytorch_worker_seed() + self.epoch)
        else:
            seed = utils.make_seed(utils.pytorch_worker_seed(), self.epoch, os.getpid(), time.time_ns(), os.urandom(4))

        rng = random.Random(seed)

        for _ in range(len(self.anno_list)):
            index = rng.randint(0, len(self.anno_list) - 1)

            yield {"anno": self.anno_list[index], "index": index}

    def sample_anno(self):
        while True:
            for anno in self._sample_anno():
                yield anno

    def _sample_data(self, data_info):
        # load images
        img_bytes = self.client.get(data_info["img_path"])
        assert img_bytes is not None, f"Failed to load image {data_info['img_path']}"
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        images = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        # preprocessing
        output = {}
        output["pixel_values"] = self.transforms(Image.fromarray(np.uint8(images)))

        if data_info["task_type"] == "outpainting":
            temp_mask = torch.zeros((self.resolution, self.resolution))
            mask_rp = random.random()
            mask_lp = random.random()
            mask_tp = random.random()
            mask_bp = random.random()
            if mask_rp <= 0.5 and mask_lp <= 0.5 and mask_tp <= 0.5 and mask_bp <= 0.5:
                mask_bp = 1
                mask_tp = 1
                mask_lp = 1
                mask_rp = 1
            if mask_rp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.resolution / 2 * cur_p) - 1, 0)
                temp_mask[:, self.resolution - 1 - mask_len :] = 1
            if mask_lp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.resolution / 2 * cur_p), 0)
                temp_mask[:, :mask_len] = 1
            if mask_bp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.resolution / 2 * cur_p) - 1, 0)
                temp_mask[self.resolution - 1 - mask_len :, :] = 1
            if mask_tp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.resolution / 2 * cur_p), 0)
                temp_mask[:mask_len, :] = 1
            output["mask"] = temp_mask.unsqueeze(0)

        elif data_info["task_type"] == "inpainting":
            mask_image = Image.open(data_info["mask"]).convert("L")
            if random.random() > 0.5:
                mask_image = mask_image.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                mask_image = mask_image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask_image.resize((self.resolution, self.resolution), Image.LANCZOS)
            mask = np.array(mask)
            mask = mask.astype(np.float32)

            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask[mask > 128] = 255
            mask[mask <= 128] = 0

            mask = Image.fromarray(mask.astype("uint8"))
            mask = transforms.ToTensor()(mask)
            mask[mask != 0] = 1
            output["mask"] = mask

        elif data_info["task_type"] == "t2i":
            output["mask"] = torch.ones((1, self.resolution, self.resolution))

        else:
            raise NotImplementedError(f"Task type {data_info['task_type']} is not implemented.")

        alpha = torch.tensor((1.0, 0.0))
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

    def sample_data(self, anno_info):
        data = self.client.get(anno_info)
        anno_str = data.decode()
        annotations = anno_str.split("\n")
        annotations = [json.loads(anno) for anno in annotations if anno != ""]
        random.shuffle(annotations)

        buffer = []
        for _, anno_info in enumerate(annotations):
            # load image and annotation data
            mask_name = ""
            prompt = ""
            promptA = self.task_prompt.context_inpainting.placeholder_tokens
            promptB = self.task_prompt.context_inpainting.placeholder_tokens
            class_p = random.random()
            task_type = ""

            if class_p < 0.25:
                # t2i: ctxt + desc
                task_type = "t2i"
                prompt = anno_info["content"]

            elif class_p < 0.5:
                # outpainting: ctxt + NULL
                task_type = "outpainting"

            else:
                # inpainting: ctxt + desc or NULL
                task_type = "inpainting"
                if random.random() < 0.2:
                    prompt = anno_info["content"]
                mask_name = random.choice(self.random_mask_list)

            if self.desc_prefix and prompt != "":  # for unet-based models
                promptA, promptB = f"{promptA} {prompt}", f"{promptB} {prompt}"

            # 10% probability to drop all conditions for unconditional generation
            # NULL + NULL
            if random.random() < 0.1:
                promptA = promptB = prompt = ""

            remark = anno_info["remark"]
            aesthetic_score = remark["aesthetic_score"]
            clip_score = remark["similarity"]
            o_height = remark["height"]
            o_width = remark["width"]
            o_pwatermark = remark["pwatermark"]

            img_list = anno_info["img_list"]
            if not img_list:
                continue

            img_info = img_list[list(img_list.keys())[0]]
            if not img_info["jpg_exists"]:
                continue

            if aesthetic_score is None or aesthetic_score < self.aesthetic_score_threshold:
                continue

            if clip_score is None or (
                self.clip_score_threshold is not None and clip_score < self.clip_score_threshold
            ):
                continue

            if o_height is None or o_width is None:
                continue
            if o_height < 512 or o_width < 512:
                continue
            if o_pwatermark is None or o_pwatermark > 0.5:
                continue

            jpg_path = img_info["jpg_path"]
            jpg_path = jpg_path[1:] if jpg_path.startswith("/") else jpg_path
            img_path = osp.join(self.client_prefix + img_info["jpg_prefix"], jpg_path)
            data_info = {
                "img_path": img_path,
                "mask": mask_name,
                "promptA": promptA,
                "promptB": promptB,
                "prompt": prompt,
                "task_type": task_type,
            }

            if self.bufsize is None:
                try:
                    yield self._sample_data(data_info)
                except Exception:
                    logger.info(f"Error in {data_info}")
                    continue

            elif len(buffer) < self.bufsize:
                buffer.append(data_info)

            else:
                select_idx = random.randint(0, self.bufsize - 1)

                selected_data = buffer[select_idx]
                buffer[select_idx] = data_info

                try:
                    data = self._sample_data(selected_data)
                    yield data
                except Exception:
                    logger.info(f"Error in {selected_data}")
                    continue

        for data_info in buffer:
            try:
                yield self._sample_data(data_info)
            except Exception:
                logger.info(f"Error in {data_info}")
                continue

    def __iter__(self):
        for anno_info in self.sample_anno():
            for data in self.sample_data(anno_info["anno"]):
                yield data

    def __len__(self):
        return 999_999_999

    def __repr__(self):
        return f"LaionIterJsonDataset(anno_root={self.anno_root}, random_mask_root={self.random_mask_root})"
