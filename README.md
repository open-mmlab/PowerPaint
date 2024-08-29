# 🖌️ ECCV 2024 | PowerPaint: A Versatile Image Inpainting Model

[**A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting**](https://arxiv.org/abs/2312.03594)

[Junhao Zhuang](https://github.com/zhuang2002), [Yanhong Zeng](https://zengyh1900.github.io/), [Wenran Liu](https://github.com/liuwenran), [Chun Yuan†](https://www.sigs.tsinghua.edu.cn/yc2_en/main.htm), [Kai Chen†](https://chenkai.site/)

(†corresponding author)

[![arXiv](https://img.shields.io/badge/arXiv-2312.03594-b31b1b.svg)](https://arxiv.org/abs/2312.03594)
[![Project Page](https://img.shields.io/badge/PowerPaint-Website-green)](https://powerpaint.github.io/)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/rangoliu/PowerPaint)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/JunhaoZhuang/PowerPaint-v1)

**Your star means a lot for us to develop this project!** :star:

PowerPaint is a high-quality versatile image inpainting model that supports text-guided object inpainting, object removal, shape-guided object insertion, and outpainting at the same time. We achieve this by learning with tailored task prompts for different inpainting tasks.

<img src='https://github.com/open-mmlab/mmagic/assets/12782558/acd01391-c73f-4997-aafd-0869aebcc915'/>


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

## Get Started

```bash
# Clone the Repository
git clone git@github.com:open-mmlab/PowerPaint.git

# Create Virtual Environment with Conda
conda create --name ppt python=3.9
conda activate ppt

# Install Dependencies
pip install -r requirements/requirements.txt
```

Or you can construct a conda environment from scratch by running the following command:

```bash
conda env create -f requirements/ppt.yaml
conda activate ppt
```

## Inference

You can launch the Gradio interface for PowerPaint by running the following command:

```bash
# Set up Git LFS
conda install git-lfs
git lfs install

# Clone PowerPaint Model
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v1/ ./checkpoints/ppt-v1

python app.py --share
```

For the BrushNet-based PowerPaint, you can run the following command:
```bash
# Clone PowerPaint Model
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint_v2/ ./checkpoints/ppt-v2

python app.py --share --version ppt-v2 --checkpoint_dir checkpoints/ppt-v2
```

### Text-Guided Object Inpainting

After launching the Gradio interface, you can insert objects into images by uploading your image, drawing the mask, selecting the tab of `Text-guided object inpainting` and inputting the text prompt. The model will then generate the output image.

|Input|Output|
|---------------|-----------------|
| <img src="assets/gradio_text_objinpaint.jpg"> | <img src="assets/gradio_text_objinpaint_result.jpg">



### Text-Guided Object Inpainting with ControlNet

Fortunately, PowerPaint is compatible with ControlNet. Therefore, users can generate object with a control image.

|Input| Condition | Control Image |Output|
|-------|--------|-------|----------|
|<img src="assets/control_input.jpg"> | Canny| <img src="assets/canny.jpg"> | <img src="assets/canny_result.jpg">
|<img src="assets/control_input.jpg"> | Depth| <img src="assets/depth.jpg"> | <img src="assets/depth_result.jpg">
|<img src="assets/control_input.jpg"> | HED| <img src="assets/hed.jpg"> | <img src="assets/hed_result.jpg">
|<img src="assets/pose_input.jpg"> | Pose| <img src="assets/pose_control.jpg"> | <img src="assets/pose_result.jpg">


### Object Removal

For object removal, you need to select the tab of `Object removal inpainting` and you don't need to input any prompts. PowerPaint is able to fill in the masked region according to context background.

We remain the text box for inputing prompt, allowing users to further suppress object generation by using negative prompts.
Specifically, we recommend to use 10 or higher value for Guidance Scale. If undesired objects appear in the masked area, you can address this by specifically increasing the Guidance Scale.

|Input|Output|
|---------------|-----------------|
| <img src="assets/gradio_objremoval.jpg"> | <img src="assets/gradio_objremoval_result.jpg">



### Image Outpainting

For image outpainting, you don't need to input any text prompt. You can simply select the tab of `Image outpainting` and adjust the slider for `horizontal expansion ratio` and `vertical expansion ratio`, then PowerPaint will extend the image for you.

|Input|Output|
|---------------|-----------------|
| <img src="assets/gradio_outpaint.jpg"> | <img src="assets/gradio_outpaint_result.jpg">



### Shape-Guided Object Inpainting

PowerPaint also supports shape-guided object inpainting, which allows users to control the fitting degree of the generated objects to the shape of masks. You can select the tab of `Shape-guided object inpainting` and input the text prompt. Then, you can adjust the slider of `fitting degree` to control the shape of generated object.

Taking the following cases as example, you can draw a square mask and use a high fitting degree, e.g., 0.95, to generate a bread to fit in the mask shape. For the same mask, you can also use a low fitting degree, e.g., 0.55, to generate a reasonable result for rabbit. However, if you use a high fitting degree for the 'square rabit', the result may look funny.

Basically, we recommend to use 0.5-0.6 for fitting degree when you want to generate objects that are not constrained by the mask shape. If you want to generate objects that fit the mask shape, you can use 0.8-0.95 for fitting degree.


|Prompt | Fitting Degree | Input| Output|
|-------|--------|--------|---------|
|a bread  | 0.95| <img src="assets/shapeguided_s1.jpg"> | <img src="assets/shapeguided_s1_result.jpg">
|a rabbit | 0.55| <img src="assets/shapeguided_s1_rabbit.jpg"> | <img src="assets/shapeguided_s1_rabbit_result.jpg">
|a rabbit | 0.95|<img src="assets/shapeguided_s1_rabbit_high.jpg"> | <img src="assets/shapeguided_s1_rabbit_high_result.jpg">
|a rabbit | 0.95 | <img src="assets/accurate_rabbit.jpg"> | <img src="assets/accurate_rabbit_result.jpg">



## Training

Since we refactored the code for better support of both the U-Net-based and Brush-Net network architectures, please refer to our [`dev` branch](https://github.com/open-mmlab/PowerPaint/tree/dev?tab=readme-ov-file#training) for more details on training.

If you have any issues, please feel free to [open issues](https://github.com/open-mmlab/PowerPaint/issues). Any [pull request](https://github.com/open-mmlab/PowerPaint/pulls) are welcome and we will review them asap.



## Contact Us

**Junhao Zhuang**: zhuangjh23@mails.tsinghua.edu.cn

**Yanhong Zeng**: zengyh1900@gmail.com




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
