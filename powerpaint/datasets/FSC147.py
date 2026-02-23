import json
import os
import random

import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


logger = get_logger(__name__)

def load_data(data_path):
    data_info_list = []
    for img_folder in os.listdir(data_path):
        try:
            # load data
            with open(os.path.join(data_path, img_folder, 'annotation.json'), 'r') as f:
                anno = json.load(f)
            mask_bbox = anno['inpainted_bboxes'][0]
            loc_bbox = anno['inpainted_bboxes'][1]
            prompt = anno['class_based_caption']
            img_path = os.path.join(data_path, img_folder, 'ground_truth.jpg')
            data_info = {
                "img_path": img_path,
                "mask_bbox": mask_bbox,
                "loc_bbox": loc_bbox,
                "prompt": prompt
            }
            data_info_list.append(data_info)
        except:
            continue

    return data_info_list

def augment_images(img_path, mask_bbox, loc_bbox, resolution):
    """
    Crop và resize img về kích thước resolution, thay đổi tọa độ bbox tương ứng.
    Đảm bảo crop vào phần có mask_bbox và loc_bbox
    parameters:
        img_path: path to img
        mask_bbox, loc_bbox: (x1, y1, x2, y2) theo tọa độ ảnh gốc
    return:
    img: numpy.ndarray [resolution x resolution x 3]
    mask: numpy.ndarray [resolution x resolution]
    loc_bbox: [x,y,x,y]
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Kích thước hình vuông crop
    square_size = min(h, w)
    half = square_size // 2

    # Hai bbox cần đảm bảo nằm trong crop
    bboxes = np.array([mask_bbox, loc_bbox])

    # Tìm min/max của toàn bộ vùng cần cover
    x_min = np.min(bboxes[:, 0])
    y_min = np.min(bboxes[:, 1])
    x_max = np.max(bboxes[:, 2])
    y_max = np.max(bboxes[:, 3])

    # Tâm ban đầu
    cx = (x_min + x_max)/2
    cy = (y_min + y_max)/2

    # Crop ban đầu
    crop_x1 = int(cx - half)
    crop_y1 = int(cy-half)
    crop_x2 = crop_x1 + square_size
    crop_y2 = crop_y1 + square_size

    # Hàm dịch crop vào trong ảnh
    def shift_into_image(x1, y1, x2, y2, W, H):
        dx1 = max(0, -x1)
        dy1 = max(0, -y1)
        dx2 = max(0, x2 - W)
        dy2 = max(0, y2 - H)
        return x1 + dx1 - dx2, y1 + dy1 - dy2, x2 + dx1 - dx2, y2 + dy1 - dy2

    crop_x1, crop_y1, crop_x2, crop_y2 = shift_into_image(
        crop_x1, crop_y1, crop_x2, crop_y2, w, h
    )

    # Kiểm tra đảm bảo chứa đủ bbox
    for (bx1, by1, bx2, by2) in bboxes:
        if not (crop_x1 <= bx1 and bx2 <= crop_x2 and crop_y1 <= by1 and by2 <= crop_y2):
            logger.info(f"{img_path} Bbox quá to, không thể chứa trong crop hình vuông min(h,w).")

    # Crop
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize
    resized = cv2.resize(crop, (resolution, resolution))
    scale = resolution/square_size

    # Chuyển toạ độ bbox
    def convert_bbox(bbox):
        x1, y1, x2, y2 = bbox
        x1 = int((x1 - crop_x1) * scale)
        y1 = int((y1 - crop_y1) * scale)
        x2 = int((x2 - crop_x1) * scale)
        y2 = int((y2 - crop_y1) * scale)
        return [x1, y1, x2, y2]

    new_mask_bbox = convert_bbox(mask_bbox)
    new_loc_bbox = convert_bbox(loc_bbox)

    # Tạo mask
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    x1, y1, x2, y2 = new_mask_bbox
    mask[y1:y2, x1:x2] = 255


    return resized, mask, new_loc_bbox


class FSCDataset(Dataset):
    """
    Dataset class for Fine-tuning PowerPaint on custom local data.
    """
    def __init__(
        self,
        data_path,
        transforms,
        pipeline,
        task_prompt,
        resolution
    ):
        self.transforms = transforms
        self.pipeline = pipeline
        self.task_prompt = task_prompt
        self.resolution = resolution

        self.data_info_list = load_data(data_path)


    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):
        # Cơ chế retry: Nếu load lỗi ảnh này, tự động lấy ảnh ngẫu nhiên khác
        # try:
        return self._get_item_inner(idx)
        # except Exception as e:
        #     logger.info(f"Error loading index {idx}: {e}. Retrying with random index...")
        #     return self.__getitem__(random.randint(0, len(self) - 1))

    def _get_item_inner(self, idx):
        data_info = self.data_info_list[idx]

        output = {}
        img, mask, loc_bbox = augment_images(data_info['img_path'], data_info['mask_bbox'], data_info['loc_bbox'], self.resolution)
        ToT = transforms.ToTensor()
        if self.transforms:
            img = Image.fromarray(img).convert('RGB')
            output["pixel_values"] = self.transforms(img)
        else:
            img = Image.fromarray(img).convert('RGB')

            # convert to tensors
            img = ToT(img)

            # normalize the image with mean and std
            normalize = transforms.Normalize(mean=[0.5], std=[0.5])
            img = normalize(img)

            output["pixel_values"] = img

        mask = Image.fromarray(mask).convert('L')
        mask = ToT(mask)
        mask[mask != 0] = 1
        output["mask"] = mask



        if random.random() < 0.3:
            prompt = ""
        else:
            prompt = data_info['prompt']

        promptA = self.task_prompt.indomain_inpainting.placeholder_tokens
        promptB = self.task_prompt.indomain_inpainting.placeholder_tokens
        promptA, promptB = f"{promptA} {prompt}", f"{promptB} {prompt}"


        prompt = self.pipeline.maybe_convert_prompt(prompt, self.pipeline.tokenizer)
        promptA = self.pipeline.maybe_convert_prompt(promptA, self.pipeline.tokenizer)
        promptB = self.pipeline.maybe_convert_prompt(promptB, self.pipeline.tokenizer)
        output["input_idsA"], output["input_idsB"], output["input_ids"] = self.pipeline.tokenizer(
            [promptA, promptB, prompt],
            max_length=self.pipeline.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        alpha = torch.tensor((1.0, 0.0))
        output["tradeoff"] = alpha

        return output

