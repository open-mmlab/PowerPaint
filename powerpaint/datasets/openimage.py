import os
import random
import time

import cv2
import numpy as np
import torch
from petrel_client.client import Client
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms

from .utils import save_log


def random_warponly(img, sigma=15, patch=40):  # 15
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
    # print(np.shape(mask))
    H = np.shape(mask)[0]
    W = np.shape(mask)[1]
    # 获取掩码中非零元素的索引
    nonzero_indices = np.nonzero(mask)
    if len(nonzero_indices) == 0:
        return mask
    # 获取最小边界框的左上角和右下角坐标
    min_row = max(np.min(nonzero_indices[0]) - pp, 0)
    max_row = min(np.max(nonzero_indices[0]) + pp, H)
    min_col = max(np.min(nonzero_indices[1]) - pp, 0)
    max_col = min(np.max(nonzero_indices[1]) + pp, W)
    # 创建最小边界框
    bounding_box = np.zeros_like(mask)
    bounding_box[min_row : max_row + 1, min_col : max_col + 1] = 255
    return bounding_box


def pool_num(img, pool_step=10):  # 15
    warped = img
    # print(np.max(warped))
    warped[warped > 0.5] = 1.0
    warped[warped <= 0.5] = 0.0

    rand_it = pool_step
    rand_it = int(max(0, rand_it))
    kernel = np.ones((3, 3))
    if rand_it != 0:
        warped = cv2.dilate(src=warped, kernel=kernel, iterations=rand_it)

    warped = warped
    return warped


def random_warp(img, sigma=15, patch=60, pool_step=10):  # 15
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

    rand_it = pool_step
    rand_it = int(max(0, rand_it))
    kernel = np.ones((3, 3))
    if rand_it != 0:
        warped = cv2.dilate(src=warped, kernel=kernel, iterations=rand_it)

    warped = warped * 255.0
    return warped


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
    def __init__(
        self,
        pipeline,
        bufsize=None,
        clip_score_threshold=None,
        aesthetic_score_threshold=0.5,
        deterministic=False,
        transforms=None,
        args=None,
        data_tokenizer=None,
    ):
        super().__init__()
        # for data loading
        self.client = Client(enable_multi_cluster=True)

        # for data shuffle
        self.bufsize = bufsize

        # for data filter
        self.aesthetic_score_threshold = aesthetic_score_threshold
        self.clip_score_threshold = clip_score_threshold

        # for data pipeline
        self.pipeline = Compose(pipeline)

        # for shard sampler
        self.epoch = -1
        self.deterministic = deterministic
        self.args = args
        self.transforms = transforms
        self.tokenizer = data_tokenizer
        # self.tokenizer = CLIPTokenizer.from_pretrained(
        # self.args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None #self.args.pretrained_model_name_or_path
        # )

    def augment_images(self, image, mask):
        mask[mask > 128] = 255
        mask[mask <= 128] = 0
        mask = Image.fromarray(mask.astype("uint8"))
        resize = transforms.Resize((self.args.resolution))  # self.args.resolution
        image = resize(image)
        mask = resize(mask)

        crop = RandomCrop(self.args.resolution)
        image, mask = crop(image, mask)
        # crop = transforms.CenterCrop((512))
        # image = crop(image)
        # mask = crop(mask)

        # 50% chance of applying horizontal flip
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # convert the image and mask to tensors
        toT = transforms.ToTensor()
        image = toT(image)
        mask = toT(mask)
        mask[mask != 0] = 1
        # print(type(mask))
        # mask = transforms.functional.pil_to_tensor(mask)
        # print(mask)

        # normalize the image with mean and std
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize(image)

        return image.unsqueeze(0), mask.unsqueeze(0)

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
        return inputs.input_ids

    def preprocess_train(self, image, mask):
        images = image["img"]
        output = {}
        O_image = Image.fromarray(np.uint8(images))
        width, height = O_image.size
        mask = mask.resize((width, height), Image.ANTIALIAS)

        rate = random.random()
        mask = np.array(mask)
        mask = mask.astype(np.float32)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        S_mask = len(np.nonzero(mask)[0])
        if S_mask == 0:
            alpha = torch.tensor((1.0, 0.0)).unsqueeze(0)
            aug_mask = mask
        else:
            rrpp = random.random()
            if image["lama_flag"]:
                aug_mask = mask
            elif image["bbox_flag"]:
                aug_mask = get_min_bounding_box(mask, pp=2)
                if random.random() > 0.5:
                    aug_mask = random_warponly(
                        aug_mask, sigma=20 / 200 * (S_mask ** (0.5)), patch=max(60 / 200 * (S_mask ** (0.5)), 4)
                    )
            else:
                aug_mask = mask
            S_aug = len(np.nonzero(aug_mask)[0])
            Srate = S_mask / S_aug
            Srate = max(Srate, 0)
            Srate = min(Srate, 1)
            if image["remove_flag"]:
                alpha = torch.tensor((1.0, 0.0)).unsqueeze(0)
            else:
                alpha = torch.tensor((Srate, 1 - Srate)).unsqueeze(0)

        output["pixel_values"], output["mask"] = self.augment_images(O_image, aug_mask)
        output["input_idsA"] = self.tokenize_captions(image["promptA"])
        output["input_idsB"] = self.tokenize_captions(image["promptB"])
        output["input_idsC"] = self.tokenize_captions(image["promptC"])
        output["input_ids"] = self.tokenize_captions(image["prompt"])
        output["tradeoff"] = torch.tensor((1.0, 0.0)).unsqueeze(0)

        return output

    def sample_data(self):
        start_it = time.time()
        dl_end_it = time.time()

        lama_root = "/mnt/petrelfs/zhuangjunhao/code/inpainting_mask/random_mask/"
        lama_list = os.listdir(lama_root)
        self.lama_len = len(lama_list)

        load_end_it = time.time()

        save_log(f"[Anno Download] {dl_end_it - start_it:.3f}s")
        save_log(f"[Anno Load] {load_end_it - dl_end_it:.3f}s")

        buffer = []

        for idx in range(self.total_num):
            # data_end_it = time.time()
            bbox_flag = 0
            lama_flag = 0
            ap = random.random()
            if ap > 1:
                remove_flag = 1
                if random.random() > 0.9:
                    bbox_flag = 1
                anno_info = self.annotations[idx]
                anno_info = anno_info.split(",")
                rand_idx = random.randint(1, self.total_num - 2)
                anno_info_rand = self.annotations[rand_idx]
                anno_info_rand = anno_info_rand.split(",")
                mask_name = anno_info_rand[2]

                if random.random() > 0.1:
                    rand_idx = random.randint(1, self.lama_len - 2)
                    mask_name = "/" + lama_list[rand_idx]
                    lama_flag = 1
                image_name = anno_info[0]
                promptA = "P_ctxt"
                promptB = "P_ctxt"
                if random.random() < 0.1:
                    promptA = ""
                    promptB = ""
            else:
                remove_flag = 0
                anno_info = self.annotations[idx]
                anno_info = anno_info.split(",")
                inp = random.random()
                if inp < 0.5:  # 0.5
                    if (
                        anno_info[3]
                        == "a 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911"
                    ):
                        anno_info[3] = ""
                    promptA = "P_shape"
                    promptB = "P_shape"
                    prompt = anno_info[3]

                else:
                    remove_flag = 1
                    bbox_flag = 1
                    if (
                        anno_info[3]
                        == "a 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911 1911"
                    ):
                        anno_info[3] = ""
                    promptA = "P_obj"
                    promptB = "P_obj"
                    prompt = anno_info[3]

                mask_name = anno_info[2]
                image_name = anno_info[0]

                if random.random() < 0.1:
                    promptA = ""
                    promptB = ""
                    prompt = ""

            image_info = dict(
                img_path=image_name,
                mask=mask_name,
                promptA=promptA,
                promptB=promptB,
                prompt=prompt,
                promptC="P_abc",
                remove_flag=remove_flag,
                bbox_flag=bbox_flag,
                lama_flag=lama_flag,
            )

            if self.bufsize is None:
                try:
                    if image_info["lama_flag"]:
                        mask_image = Image.open(lama_root + image_info["mask"])
                        if random.random() > 0.5:
                            mask_image = mask_image.transpose(Image.FLIP_LEFT_RIGHT)
                        if random.random() > 0.5:
                            mask_image = mask_image.transpose(Image.FLIP_TOP_BOTTOM)
                    else:
                        mask_image = Image.open(
                            "/mnt/petrelfs/zhuangjunhao/code/openimagev6/mask" + image_info["mask"]
                        )
                    image = self.pipeline(image_info)
                    s1, s2 = Image.fromarray(np.uint8(image["img"])).size
                    if s1 < 512 or s2 < 512:
                        continue

                    yield self.preprocess_train(image, mask_image)

                except Exception:
                    print(f"Error in {image_info}")
                    continue

            elif len(buffer) < self.bufsize:
                buffer.append(image_info)

            else:
                select_idx = random.randint(0, self.bufsize - 1)

                save_log(f"[Data] Idx/SelectIdx: {idx}/{select_idx} " f"Name: {image_info}")

                selected_data = buffer[select_idx]
                pipe_start_it = time.time()

                try:
                    mask_image = Image.open("/mnt/petrelfs/zhuangjunhao/code/openimagev6/mask" + selected_data["mask"])
                    image = self.pipeline(selected_data)
                    yield self.preprocess_train(image, mask_image)

                except Exception:
                    print(f"Error in {selected_data}")
                    continue

                pipe_end_it = time.time()
                save_log(f"[Pipe] {pipe_end_it - pipe_start_it:.3f}s")
                buffer[select_idx] = image_info

        for image_info in buffer:
            try:
                mask_image = Image.open("/mnt/petrelfs/zhuangjunhao/code/openimagev6/mask" + image_info["mask"])
                image = self.pipeline(image_info)
                yield self.preprocess_train(image, mask_image)
            except Exception:
                print(f"Error in {image_info}")
                continue

    def __iter__(self):
        self.annotations = []
        for i in range(16):
            with open(
                "/mnt/petrelfs/zhuangjunhao/code/Smartbrush/seg_anno/prompt_anno_" + str(i) + ".txt",
                "r",
                encoding="utf-8",
            ) as f:  # 打开文本
                data = f.read()  # 读取文本
            f.close()
            annotation = data.split("\n")
            self.annotations += annotation[:-1]

        self.total_num = len(self.annotations)
        random.shuffle(self.annotations)
        for data in self.sample_data():
            yield data

    def __len__(self):
        return 999_999_999
