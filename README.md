# A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting


### [Project Page](https://powerpaint.github.io/) | [Paper](https://arxiv.org/abs/2312.03594) | [Online Demo(OpenXlab)](https://openxlab.org.cn/apps/detail/rangoliu/PowerPaint#basic-information)

This README provides a step-by-step guide to download the repository, set up the required virtual environment named "PowerPaint" using conda, and run PowerPaint with or without ControlNet.

**Feel free to try it and give it a star!**:star:

## 🚀 News

**May 22, 2024**:fire:

- We have open-sourced the model weights for PowerPaint v2-1, rectifying some existing issues that were present during the training process of version 2. [![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1)

**April 7, 2024**:fire:

- We open source the model weights and code for PowerPaint v2. [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/zhuangjunhao/PowerPaint_v2) [![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/JunhaoZhuang/PowerPaint_v2)

**April 6, 2024**:

- We have retrained a new PowerPaint, taking inspiration from Brushnet. The [Online Demo](https://openxlab.org.cn/apps/detail/rangoliu/PowerPaint) has been updated accordingly. **We plan to release the model weights and code as open source in the next few days**.
- Tips: We preserve the cross-attention layer that was deleted by BrushNet for the task prompts input.

|  | Object insertion | Object Removal|Shape-guided Object Insertion|Outpainting|
|-----------------|-----------------|-----------------|-----------------|-----------------|
| Original Image| ![cropinput](https://github.com/Sanster/IOPaint/assets/108931120/bf91a1e8-8eaf-4be6-b47d-b8e43c9d182a)|![cropinput](https://github.com/Sanster/IOPaint/assets/108931120/c7e56119-aa57-4761-b6aa-56f8a0b72456)|![image](https://github.com/Sanster/IOPaint/assets/108931120/cbbfe84e-2bf1-425b-8349-f7874f2e978c)|![cropinput](https://github.com/Sanster/IOPaint/assets/108931120/134bb707-0fe5-4d22-a0ca-d440fa521365)|
| Output| ![image](https://github.com/Sanster/IOPaint/assets/108931120/ee777506-d336-4275-94f6-31abf9521866)| ![image](https://github.com/Sanster/IOPaint/assets/108931120/e9d8cf6c-13b8-443c-b327-6f27da54cda6)|![image](https://github.com/Sanster/IOPaint/assets/108931120/cc3008c9-37dd-4d98-ad43-58f67be872dc)|![image](https://github.com/Sanster/IOPaint/assets/108931120/18d8ca23-e6d7-4680-977f-e66341312476)|

**December 22, 2023**:wrench:

- The logical error in loading ControlNet has been rectified. The `gradio_PowerPaint.py` file and [Online Demo](https://openxlab.org.cn/apps/detail/rangoliu/PowerPaint) have also been updated.

**December 18, 2023**

*Enhanced PowerPaint Model*

- We are delighted to announce the release of more stable model weights. These refined weights can now be accessed on [Hugging Face](https://huggingface.co/JunhaoZhuang/PowerPaint-v1/tree/main). The `gradio_PowerPaint.py` file and [Online Demo](https://openxlab.org.cn/apps/detail/rangoliu/PowerPaint) have also been updated as part of this release.



________________
<img src='https://github.com/open-mmlab/mmagic/assets/12782558/acd01391-c73f-4997-aafd-0869aebcc915'/>

## Getting Started

```bash
# Clone the Repository
git clone https://github.com/zhuang2002/PowerPaint.git

# Navigate to the Repository
cd projects/powerpaint

# Create Virtual Environment with Conda
conda create --name PowerPaint python=3.9
conda activate PowerPaint

# Install Dependencies
pip install -r requirements.txt
```
## PowerPaint v2

```bash
python gradio_PowerPaint_BrushNet.py
```

## PowerPaint v1

```bash
# Create Models Folder
mkdir models

# Set up Git LFS
git lfs install

# Clone PowerPaint Model
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v1/ ./models

python gradio_PowerPaint.py
```

This command will launch the Gradio interface for PowerPaint.

Feel free to explore and edit images with PowerPaint!

## BibTeX

```
@misc{zhuang2023task,
      title={A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting},
      author={Junhao Zhuang and Yanhong Zeng and Wenran Liu and Chun Yuan and Kai Chen},
      year={2023},
      eprint={2312.03594},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
