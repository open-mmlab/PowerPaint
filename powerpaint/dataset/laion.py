import json
import os
import os.path as osp
import random
import time

# from mmcv.transforms import CopyKey
import numpy as np
import torch
from mmagic.registry import DATASETS
from mmengine.dataset import Compose
from petrel_client.client import Client
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from webdataset import utils


# cv2.setNumThreads(0)
SUPPORTED_SUFFIX = [
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "png",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic",
]
SAVE_LOG = os.environ.get("SAVE_LOG", False)
LOG_DIR = os.environ.get("LOG_DIR", "./logs")


def my_copy_key(data):
    # 如果输入字典中存在'prompt'键，就将其复制到输出字典中
    if "prompt" in data:
        data["prompt"] = data["prompt"]
    return data


def save_log(log):
    if not SAVE_LOG:
        return

    worker_info = get_worker_info()
    worker_id = 0 if worker_info is None else worker_info.id
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(osp.join(LOG_DIR, f"worker_{worker_id}.log"), "a") as f:
        f.write(log)
        f.write("\n")


@DATASETS.register_module()
class LaionIterJsonDataset(IterableDataset):
    def __init__(
        self,
        anno_root,
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
        self.anno_list = self.load_anno_list(anno_root)

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
        self.transforms = transforms
        self.args = args
        self.tokenizer = data_tokenizer

    def tokenize_captions(self, examples, is_train=True):
        captions = []
        caption = examples
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(f"Caption column `{'prompt'}` should contain either strings or lists of strings.")
        # print(captions)
        # for caption in examples['prompt']:
        #     print(caption)
        #     if isinstance(caption, str):
        #         captions.append(caption)
        #     elif isinstance(caption, (list, np.ndarray)):
        #         # take a random caption if there are multiple
        #         captions.append(random.choice(caption) if is_train else caption[0])
        #     else:
        #         raise ValueError(
        #             f"Caption column `{'prompt'}` should contain either strings or lists of strings."
        #         )
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def load_anno_list(self, anno_root):
        anno_list = []
        for anno in self.client.list(anno_root):
            if not anno.endswith(".jsonl"):
                continue
            anno_list.append(os.path.join(anno_root, anno))
            # print(anno)

        return anno_list

    def add_prompt(self, ids):
        last_idx = torch.nonzero(ids["attention_mask"])[-1, 1]
        if last_idx < (self.tokenizer.model_max_length - 10):
            ids["attention_mask"][0, last_idx + 1 : last_idx + 11] = 1
            ids["input_ids"][0, last_idx : last_idx + 10] = torch.tensor(range(49408, 49418))
            ids["input_ids"][0, last_idx + 10] = 49407
        else:
            ids["attention_mask"][0, :] = 1
            ids["input_ids"][0, self.tokenizer.model_max_length - 11 : self.tokenizer.model_max_length - 1] = (
                torch.tensor(range(49408, 49418))
            )
            ids["input_ids"][0, self.tokenizer.model_max_length - 1] = 49407
        return ids

    def preprocess_train(self, examples):
        images = examples["img"]
        output = {}
        output["pixel_values"] = self.transforms(Image.fromarray(np.uint8(images))).unsqueeze(0)
        # flag_null1 = examples["flag_null"]
        outpaint_p = examples["outpaint_flag"]
        inpaint_flag1 = examples["inpaint_flag"]
        T2I_flag1 = examples["T2I_flag"]
        output["input_idsA"] = self.tokenize_captions(examples["promptA"]).input_ids
        output["input_idsB"] = self.tokenize_captions(examples["promptB"]).input_ids
        output["input_ids"] = self.tokenize_captions(examples["prompt"]).input_ids

        output["input_idsC"] = self.tokenize_captions(examples["promptC"]).input_ids

        # print(T2I_flag1,inpaint_flag1,outpaint_p,flag_null1,output["input_idsA"])

        if outpaint_p:
            temp_mask = torch.zeros((1, 1, self.args.resolution, self.args.resolution))
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
                mask_len = max(int(self.args.resolution / 2 * cur_p) - 1, 0)
                temp_mask[:, :, :, self.args.resolution - 1 - mask_len :] = 1
            if mask_lp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.args.resolution / 2 * cur_p), 0)
                temp_mask[:, :, :, :mask_len] = 1
            if mask_bp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.args.resolution / 2 * cur_p) - 1, 0)
                temp_mask[:, :, self.args.resolution - 1 - mask_len :, :] = 1
            if mask_tp > 0.5:
                cur_p = random.random()
                mask_len = max(int(self.args.resolution / 2 * cur_p), 0)
                temp_mask[:, :, :mask_len, :] = 1
            output["mask"] = temp_mask
        if inpaint_flag1:
            mask_image = Image.open(self.lama_root + examples["mask"])
            if random.random() > 0.5:
                mask_image = mask_image.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                mask_image = mask_image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask_image.resize((self.args.resolution, self.args.resolution), Image.ANTIALIAS)
            mask = np.array(mask)
            mask = mask.astype(np.float32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            mask[mask > 128] = 255
            mask[mask <= 128] = 0
            mask = Image.fromarray(mask.astype("uint8"))
            toT = transforms.ToTensor()
            mask = toT(mask)
            mask[mask != 0] = 1
            output["mask"] = mask.unsqueeze(0)
        if T2I_flag1:
            output["mask"] = torch.ones((1, 1, self.args.resolution, self.args.resolution))
        alpha = torch.tensor((1.0, 0.0)).unsqueeze(0)
        output["tradeoff"] = alpha

        return output

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

        # data_start_it = time.time()
        # import ipdb

        # ipdb.set_trace()
        self.lama_root = "/mnt/petrelfs/zhuangjunhao/code/inpainting_mask/random_mask/"
        lama_list = os.listdir(self.lama_root)
        self.lama_len = len(lama_list)
        for idx, anno_info in enumerate(annotations):
            class_p = random.random()
            T2I_flag = 0
            outpaint_flag = 0
            flag_null = 0
            inpaint_flag = 0
            if class_p < 0.32:
                T2I_flag = 1
                promptA = ""
                promptB = ""
                prompt = anno_info["content"]
                mask_name = ""

            elif class_p < 0.52:
                outpaint_flag = 1
                promptA = "P_ctxt"
                promptB = "P_ctxt"
                prompt = ""
                mask_name = ""
            else:
                inpaint_flag = 1
                promptA = "P_ctxt"
                promptB = "P_ctxt"
                prompt = anno_info["content"]
                temp = random.random()
                if temp < 0.5:
                    prompt = ""
                rand_idx = random.randint(1, self.lama_len - 2)
                mask_name = "/" + lama_list[rand_idx]

            rp = random.random()
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

            image_id = list(img_list.keys())[0]
            img_info = img_list[image_id]

            if not img_info["jpg_exists"]:
                continue

            if aesthetic_score is None:
                continue

            if aesthetic_score < self.aesthetic_score_threshold:
                continue

            if clip_score is None:
                continue

            if self.clip_score_threshold is not None and clip_score < self.clip_score_threshold:
                continue

            if o_height is None:
                continue
            if o_width is None:
                continue
            if o_pwatermark is None:
                continue
            if o_height < 512:
                continue
            if o_width < 512:
                continue
            if o_pwatermark > 0.5:
                continue

            img_path = osp.join(img_info["jpg_prefix"], img_info["jpg_path"])
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
                try:
                    yield self.preprocess_train(self.pipeline(data_info))
                except Exception:
                    print(f"Error in {data_info}")
                    continue

            elif len(buffer) < self.bufsize:
                buffer.append(data_info)

            else:
                select_idx = random.randint(0, self.bufsize - 1)

                save_log(f"[Data] Idx/SelectIdx: {idx}/{select_idx} " f"Name: {img_path}")

                selected_data = buffer[select_idx]
                buffer[select_idx] = data_info
                pipe_start_it = time.time()

                try:
                    data = self.preprocess_train(self.pipeline(selected_data))
                    yield data

                except Exception:
                    print(f"Error in {selected_data}")
                    continue
                pipe_end_it = time.time()
                save_log(f"[Pipe] {pipe_end_it - pipe_start_it:.3f}s")
        for data_info in buffer:
            try:
                yield self.preprocess_train(self.pipeline(data_info))
            except Exception:
                print(f"Error in {data_info}")
                continue

    def __iter__(self):
        for anno_info in self.sample_anno():
            for data in self.sample_data(anno_info["anno"]):
                yield data

    def __len__(self):
        return 999_999_999
