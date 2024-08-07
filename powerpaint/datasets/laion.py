import json
import os
import os.path as osp
import random
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms
from webdataset import utils


try:
    from petrel_client.client import Client
except ImportError:
    print("Failed to import petrel_client. Please install it if you are using petrel-oss.")
from .utils import save_log


class LaionIterJsonDataset(IterableDataset):
    """Load data from Laion.
    PowerPaint mainly uses laion as training data for:
        - text-to-image generation
        - context-aware (i.e., text-free) image inpainting
        - image outpainting
    """

    def __init__(
        self,
        transforms,
        data_tokenizer,
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

            save_log(f"[Anno] Idx: {index} Name: {self.anno_list[index]}")

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
        output["pixel_values"] = self.transforms(Image.fromarray(np.uint8(images))).unsqueeze(0)
        output["input_idsA"] = self.tokenize_captions(data_info["promptA"]).input_ids
        output["input_idsB"] = self.tokenize_captions(data_info["promptB"]).input_ids
        output["input_ids"] = self.tokenize_captions(data_info["prompt"]).input_ids
        output["input_idsC"] = self.tokenize_captions(data_info["promptC"]).input_ids

        if data_info["outpaint_flag"]:
            temp_mask = torch.zeros((1, 1, self.resolution, self.resolution))
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
                temp_mask[:, :, :, self.resolution - 1 - mask_len :] = 1
            if mask_lp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.resolution / 2 * cur_p), 0)
                temp_mask[:, :, :, :mask_len] = 1
            if mask_bp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.resolution / 2 * cur_p) - 1, 0)
                temp_mask[:, :, self.resolution - 1 - mask_len :, :] = 1
            if mask_tp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.resolution / 2 * cur_p), 0)
                temp_mask[:, :, :mask_len, :] = 1
            output["mask"] = temp_mask

        elif data_info["inpaint_flag"]:
            mask_image = Image.open(data_info["mask"])
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
            output["mask"] = mask.unsqueeze(0)

        elif data_info["T2I_flag"]:
            output["mask"] = torch.ones((1, 1, self.resolution, self.resolution))

        alpha = torch.tensor((1.0, 0.0)).unsqueeze(0)
        output["tradeoff"] = alpha

        return output

    def sample_data(self, anno_info):
        start_it = time.time()
        data = self.client.get(anno_info)
        dl_end_it = time.time()

        anno_str = data.decode()
        annotations = anno_str.split("\n")
        annotations = [json.loads(anno) for anno in annotations if anno != ""]
        random.shuffle(annotations)
        load_end_it = time.time()

        save_log(f"[Anno Download] {dl_end_it - start_it:.3f}s")
        save_log(f"[Anno Load] {load_end_it - dl_end_it:.3f}s")

        buffer = []
        # load image and annotation data
        for idx, anno_info in enumerate(annotations):
            class_p = random.random()
            T2I_flag = 0
            outpaint_flag = 0
            inpaint_flag = 0
            flag_null = 0

            if class_p < 0.32:
                # text-to-image generation w/o task prompt
                T2I_flag = 1
                promptA = ""
                promptB = ""
                prompt = anno_info["content"]
                mask_name = ""

            elif class_p < 0.52:
                # image outpainting with task prompt w/o description
                outpaint_flag = 1
                promptA = "P_ctxt"
                promptB = "P_ctxt"
                prompt = ""
                mask_name = ""

            else:
                # context-aware (text-free) image inpanting w/ task prompt
                inpaint_flag = 1
                promptA = "P_ctxt"
                promptB = "P_ctxt"
                prompt = anno_info["content"]
                temp = random.random()
                # 50% probability to drop description
                if temp < 0.5:
                    prompt = ""
                mask_name = random.choice(self.random_mask_list)

            rp = random.random()
            # 10% probability to drop all conditions for unconditional generation
            if rp < 0.1:
                flag_null = 1
                promptA = ""
                promptB = ""
                prompt = ""

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

            # pdb.set_trace()
            jpg_path = img_info["jpg_path"]
            jpg_path = jpg_path[1:] if jpg_path.startswith("/") else jpg_path
            img_path = osp.join(self.client_prefix + img_info["jpg_prefix"], jpg_path)
            data_info = {
                "img_path": img_path,
                "promptA": promptA,
                "promptB": promptB,
                "prompt": prompt,
                "promptC": "P_abc",
                "outpaint_flag": outpaint_flag,
                "mask": mask_name,
                "flag_null": flag_null,
                "inpaint_flag": inpaint_flag,
                "T2I_flag": T2I_flag,
            }

            if self.bufsize is None:
                # try:
                yield self._sample_data(data_info)
                # except Exception:
                #     print(f"Error in {data_info}")
                #     continue

            elif len(buffer) < self.bufsize:
                buffer.append(data_info)

            else:
                select_idx = random.randint(0, self.bufsize - 1)

                save_log(f"[Data] Idx/SelectIdx: {idx}/{select_idx} " f"Name: {img_path}")

                selected_data = buffer[select_idx]
                buffer[select_idx] = data_info
                pipe_start_it = time.time()

                # try:
                data = self._sample_data(selected_data)
                yield data
                # except Exception:
                #     print(f"Error in {selected_data}")
                #     continue
                pipe_end_it = time.time()
                save_log(f"[Pipe] {pipe_end_it - pipe_start_it:.3f}s")

        for data_info in buffer:
            # try:
            yield self._sample_data(data_info)
            # except Exception:
            #     print(f"Error in {data_info}")
            #     continue

    def __iter__(self):
        for anno_info in self.sample_anno():
            for data in self.sample_data(anno_info["anno"]):
                yield data

    def __len__(self):
        return 999_999_999
