import argparse
import os

import cv2
import gradio as gr
import numpy as np
import torch
from accelerate.utils import set_seed
from controlnet_aux import HEDdetector, OpenposeDetector
from PIL import Image, ImageFilter
from transformers import CLIPTextModel, DPTFeatureExtractor, DPTForDepthEstimation

from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from powerpaint.models import BrushNetModel, UNet2DConditionModel
from powerpaint.pipelines import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPowerPaintBrushNetPipeline,
)


# =======================================
# use the same task prompt as training
# =======================================
TASK_LIST = ["text-guided", "object-removal", "image-outpainting", "shape-guided"]
TASK_PROMPT = {
    "ppt1": {
        "text-guided": {
            "prompt": "",
            "negative_prompt": "",
            "promptA": "P_obj {}",
            "promptB": "P_obj {}",
            "negative_promptA": "{}",
            "negative_promptB": "{}",
        },
        "object-removal": {
            "prompt": "",
            "negative_prompt": "",
            "promptA": "P_ctxt empty scene blur",
            "promptB": "P_ctxt empty scene blur",
            "negative_promptA": "P_obj {}",
            "negative_promptB": "P_obj {}",
        },
        "image-outpainting": {
            "prompt": "",
            "negative_prompt": "",
            "promptA": "P_ctxt empty scene blur, {}",
            "promptB": "P_ctxt empty scene blur, {}",
            "negative_promptA": "P_obj {}",
            "negative_promptB": "P_obj {}",
        },
        "shape-guided": {
            "prompt": "",
            "negative_prompt": "",
            "promptA": "P_shape {}",
            "promptB": "P_ctxt {}",
            "negative_promptA": "P_shape {}, worst quality, low quality, normal quality, bad quality, blurry",
            "negative_promptB": "P_ctxt {}, worst quality, low quality, normal quality, bad quality, blurry",
        },
    },
    "ppt2": {
        "text-guided": {
            "prompt": "{}",
            "negative_prompt": "{}, worst quality, low quality, normal quality, bad quality, blurry",
            "promptA": "P_obj",
            "promptB": "P_obj",
            "negative_promptA": "P_obj",
            "negative_promptB": "P_obj",
        },
        "object-removal": {
            "prompt": "{} empty scene blur",
            "negative_prompt": "{}, worst quality, low quality, normal quality, bad quality, blurry",
            "promptA": "P_ctxt",
            "promptB": "P_ctxt",
            "negative_promptA": "P_obj",
            "negative_promptB": "P_obj",
        },
        "image-outpainting": {
            "prompt": "{} empty scene blur",
            "negative_prompt": "{}, worst quality, low quality, normal quality, bad quality, blurry",
            "promptA": "P_ctxt",
            "promptB": "P_ctxt",
            "negative_promptA": "P_obj",
            "negative_promptB": "P_obj",
        },
        "shape-guided": {
            "prompt": "{}",
            "negative_prompt": "{}, worst quality, low quality, normal quality, bad quality, blurry",
            "promptA": "P_shape",
            "promptB": "P_ctxt",
            "negative_promptA": "P_shape",
            "negative_promptB": "P_ctxt",
        },
    },
}


class PowerPaintController:
    def __init__(
        self, pretrained_model_path, version, base_model_path=None, weight_dtype=torch.float16, local_files_only=False
    ) -> None:
        self.version = version
        self.pretrained_model_path = pretrained_model_path
        self.base_model_path = base_model_path
        self.local_files_only = local_files_only
        torch.set_grad_enabled(False)

        # initialize powerpaint pipeline
        if version == "ppt1":
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.base_model_path,
                unet=UNet2DConditionModel.from_pretrained(
                    self.pretrained_model_path,
                    subfolder="unet",
                    torch_dtype=weight_dtype,
                    local_files_only=local_files_only,
                ).to("cuda"),
                text_encoder=CLIPTextModel.from_pretrained(
                    self.pretrained_model_path,
                    subfolder="text_encoder",
                    torch_dtype=weight_dtype,
                    local_files_only=local_files_only,
                ).to("cuda"),
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
                safety_checker=None,
            )
        else:
            # brushnet-based version
            self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
                self.base_model_path,
                unet=UNet2DConditionModel.from_pretrained(
                    self.base_model_path,
                    subfolder="unet",
                    torch_dtype=weight_dtype,
                    local_files_only=local_files_only,
                ).to("cuda"),
                brushnet=BrushNetModel.from_pretrained(
                    self.pretrained_model_path,
                    subfolder="brushnet",
                    torch_dtype=weight_dtype,
                    local_files_only=local_files_only,
                ).to("cuda"),
                text_encoder=CLIPTextModel.from_pretrained(
                    self.pretrained_model_path,
                    subfolder="text_encoder",
                    torch_dtype=weight_dtype,
                    local_files_only=local_files_only,
                ),
                torch_dtype=weight_dtype,
                safety_checker=None,
                local_files_only=local_files_only,
            )

        # IMPORTANT:
        # 1. Add tokens in the same order and placeholder with training
        # 2. set initilize_parameters to False to avoid reinitializing the model
        self.pipe.add_tokens(
            placeholder_tokens=["P_obj", "P_ctxt", "P_shape"],
            initializer_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
            initialize_parameters=False,
        )

        self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to("cuda")

        if self.version == "ppt1":
            # initialize controlnet-related models
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
            self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

            base_control = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=local_files_only
            )
            self.control_pipe = StableDiffusionControlNetInpaintPipeline(
                self.pipe.vae,
                self.pipe.text_encoder,
                self.pipe.tokenizer,
                self.pipe.unet,
                base_control,
                self.pipe.scheduler,
                None,
                None,
                False,
            )
            self.control_pipe = self.control_pipe.to("cuda")
            self.current_control = "canny"
            # controlnet_conditioning_scale = 0.8

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    # haven't validated the controlnet part
    def load_controlnet(self, control_type):
        if self.current_control != control_type:
            if control_type == "canny" or control_type is None:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            elif control_type == "pose":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose",
                    torch_dtype=weight_dtype,
                    local_files_only=self.local_files_only,
                )
            elif control_type == "depth":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            else:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-hed", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            self.control_pipe = self.control_pipe.to("cuda")
            self.current_control = control_type

    # haven't validated the controlnet part
    def predict_controlnet(
        self,
        input_image,
        input_control_image,
        control_type,
        prompt,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        controlnet_conditioning_scale,
    ):
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
        size1, size2 = input_image["image"].convert("RGB").size

        if size1 < size2:
            input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
        else:
            input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))

        if control_type != self.current_control:
            self.load_controlnet(control_type)
        controlnet_image = input_control_image
        if control_type == "canny":
            controlnet_image = controlnet_image.resize((H, W))
            controlnet_image = np.array(controlnet_image)
            controlnet_image = cv2.Canny(controlnet_image, 100, 200)
            controlnet_image = controlnet_image[:, :, None]
            controlnet_image = np.concatenate([controlnet_image, controlnet_image, controlnet_image], axis=2)
            controlnet_image = Image.fromarray(controlnet_image)
        elif control_type == "pose":
            controlnet_image = self.openpose(controlnet_image)
        elif control_type == "depth":
            controlnet_image = controlnet_image.resize((H, W))
            controlnet_image = self.get_depth_map(controlnet_image)
        else:
            controlnet_image = self.hed(controlnet_image)

        mask_np = np.array(input_image["mask"].convert("RGB"))
        controlnet_image = controlnet_image.resize((H, W))
        set_seed(seed)
        result = self.control_pipe(
            promptA=promptB,
            promptB=promptA,
            tradeoff=1.0,
            tradeoff_nag=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            image=input_image["image"].convert("RGB"),
            mask=input_image["mask"].convert("RGB"),
            control_image=controlnet_image,
            width=H,
            height=W,
            guidance_scale=scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=ddim_steps,
        ).images[0]
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = np.array(result)
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )

        mask_np = np.array(input_image["mask"].convert("RGB"))
        m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=4))
        m_img = np.asarray(m_img) / 255.0
        img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
        ours_np = np.asarray(result) / 255.0
        ours_np = ours_np * m_img + (1 - m_img) * img_np
        result_paste = Image.fromarray(np.uint8(ours_np * 255))
        return [input_image["image"].convert("RGB"), result_paste], [controlnet_image, result_m]

    def predict(
        self,
        task,
        prompt,
        negative_prompt,
        promptA,
        negative_promptA,
        promptB,
        negative_promptB,
        fitting_degree,
        input_image,
        vertical_expansion_ratio=1,
        horizontal_expansion_ratio=1,
        ddim_steps=45,
        scale=7.5,
        seed=24,
    ):
        image, mask = input_image["image"].convert("RGB"), input_image["mask"].convert("RGB")

        # resizing images due to limited memory
        w, h = image.size
        new_size = 640 if task != "image-outpainting" else 512
        image = (
            image.resize((new_size, int(h / w * new_size)))
            if w < h
            else image.resize((int(w / h * new_size), new_size))
        )
        mask = mask.resize(image.size, Image.NEAREST)
        w, h = image.size
        hole_value = (0, 0, 0)

        # preparing masks for outpainting
        if task == "image-outpainting":
            if vertical_expansion_ratio != 1 or horizontal_expansion_ratio != 1:
                w2, h2 = int(horizontal_expansion_ratio * w), int(vertical_expansion_ratio * h)
                posw, posh = (w2 - w) // 2, (h2 - h) // 2

                new_image = Image.new("RGB", (w2, h2), hole_value)
                new_image.paste(image, (posw, posh))
                image = new_image
                new_mask = Image.new("RGB", (w2, h2), (255, 255, 255))
                new_mask.paste(mask, (posw, posh))
                mask = new_mask
                w, h = image.size

        # resizing to be divided by 8
        w, h = w // 8 * 8, h // 8 * 8
        image = image.resize((w, h))
        mask = mask.resize((w, h))
        masked_image = Image.composite(Image.new("RGB", (w, h), hole_value), image, mask.convert("L"))

        # augment mask boundary for better blending results
        # threshold = 0
        # aug_mask = mask.filter(ImageFilter.GaussianBlur(radius=5)).convert('L')
        # aug_mask = aug_mask.point(lambda p: 255 if p > threshold else 0).convert('L')
        aug_mask = mask

        result = self.pipe(
            promptA=promptA,
            promptB=promptB,
            prompt=prompt,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            negative_prompt=negative_prompt,
            tradeoff=fitting_degree,
            # input masked_image and augmented mask
            image=masked_image,
            mask=aug_mask,
            # default diffusion parameters
            num_inference_steps=ddim_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            guidance_scale=scale,
            width=w,
            height=h,
        ).images[0]

        # paste the inpainting results into original images
        result_paste = Image.composite(result, image, aug_mask.convert("L"))
        dict_out = [masked_image, result_paste]
        dict_res = [input_image["image"].convert("RGB"), input_image["mask"].convert("RGB"), result]
        return dict_out, dict_res


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--pretrained_model_path", type=str, required=True)
    args.add_argument("--base_model_path", type=str, default=None)
    args.add_argument("--weight_dtype", type=str, default="float16")
    args.add_argument("--share", action="store_true")
    args.add_argument(
        "--local_files_only", action="store_true", help="enable it to use cached files without requesting from the hub"
    )
    args.add_argument("--port", type=int, default=7860)
    args = args.parse_args()

    if os.path.exists(os.path.join(args.pretrained_model_path, "brushnet")):
        args.version = "ppt2"
    else:
        args.version = "ppt1"

    if args.base_model_path is None:
        args.base_model_path = "runwayml/stable-diffusion-v1-5"
    return args


if __name__ == "__main__":
    args = parse_args()

    # initialize the pipeline controller
    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32
    controller = PowerPaintController(
        pretrained_model_path=args.pretrained_model_path,
        version=args.version,
        base_model_path=args.base_model_path,
        weight_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )

    # ui
    with gr.Blocks(css="style.css") as demo:
        with gr.Row():
            gr.Markdown(
                "<div align='center'><font size='18'>PowerPaint: High-Quality Versatile Image Inpainting</font></div>"  # noqa
            )
        with gr.Row():
            gr.Markdown(
                "<div align='center'><font size='5'><a href='https://powerpaint.github.io/'>Project Page</a> &ensp;"  # noqa
                "<a href='https://arxiv.org/abs/2312.03594/'>Paper</a> &ensp;"
                "<a href='https://github.com/open-mmlab/powerpaint'>Code</a> </font></div>"  # noqa
            )
        with gr.Row():
            gr.Markdown(
                "**Note:** Due to network-related factors, the page may experience occasional bugs！ If the inpainting results deviate significantly from expectations, consider toggling between task options to refresh the content."  # noqa
            )

        # Attention: Due to network-related factors, the page may experience occasional bugs.
        # If the inpainting results deviate significantly from expectations,
        # consider toggling between task options to refresh the content.
        gr_task_radio = gr.Radio(TASK_LIST, value=TASK_LIST[0], show_label=False, visible=False)
        gr_prompt = {}
        gr_negative_prompt = {}
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input image and draw mask")
                input_image = gr.Image(source="upload", tool="sketch", type="pil")

                # Text-guided object inpainting
                with gr.Tab("Text-guided object inpainting") as tab_text_guided:
                    task_type = TASK_LIST[0]
                    enable_text_guided = gr.Checkbox(
                        label="Enable text-guided object inpainting", value=True, interactive=False
                    )
                    gr_prompt[task_type] = gr.Textbox(label="prompt")
                    gr_negative_prompt[task_type] = gr.Textbox(label="negative_prompt")

                    # currently, we only support controlnet in PowerPaint-v1
                    controlnet_conditioning_scale = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        value=0.5,
                        label="controlnet conditioning scale",
                        visible=args.version == "ppt1",
                    )
                    control_type = gr.Radio(
                        ["canny", "pose", "depth", "hed"], label="Control type", visible=args.version == "ppt1"
                    )
                    input_control_image = gr.Image(source="upload", type="pil", visible=args.version == "ppt1")
                tab_text_guided.select(fn=lambda: TASK_LIST[0], inputs=None, outputs=gr_task_radio)

                # Object removal inpainting
                with gr.Tab("Object removal inpainting") as tab_object_removal:
                    task_type = TASK_LIST[1]
                    enable_object_removal = gr.Checkbox(
                        label="Enable object removal inpainting",
                        value=True,
                        info="The recommended configuration for the Guidance Scale is 10 or higher. \
                        If undesired objects appear in the masked area, \
                        you can address this by specifically increasing the Guidance Scale.",
                        interactive=True,
                    )
                    gr_prompt[task_type] = gr.Textbox(label="prompt")
                    gr_negative_prompt[task_type] = gr.Textbox(label="negative_prompt")
                tab_object_removal.select(fn=lambda: TASK_LIST[1], inputs=None, outputs=gr_task_radio)

                # image outpainting
                with gr.Tab("Image outpainting") as tab_image_outpainting:
                    task_type = TASK_LIST[2]
                    enable_object_removal_outpainting = gr.Checkbox(
                        label="Enable image outpainting",
                        value=True,
                        info="The recommended configuration for the Guidance Scale is 10 or higher. \
                        If unwanted random objects appear in the extended image region, \
                            you can enhance the cleanliness of the extension area by increasing the Guidance Scale.",
                        interactive=True,
                    )
                    horizontal_expansion_ratio = gr.Slider(
                        label="horizontal expansion ratio",
                        minimum=1,
                        maximum=4,
                        step=0.05,
                        value=1,
                    )
                    vertical_expansion_ratio = gr.Slider(
                        label="vertical expansion ratio", minimum=1, maximum=4, step=0.05, value=1
                    )
                    gr_prompt[task_type] = gr.Textbox(label="Outpainting_prompt")
                    gr_negative_prompt[task_type] = gr.Textbox(label="Outpainting_negative_prompt")

                tab_image_outpainting.select(fn=lambda: TASK_LIST[2], inputs=None, outputs=gr_task_radio)

                # Shape-guided object inpainting
                with gr.Tab("Shape-guided object inpainting") as tab_shape_guided:
                    task_type = TASK_LIST[3]
                    enable_shape_guided = gr.Checkbox(
                        label="Enable shape-guided object inpainting", value=True, interactive=False
                    )
                    fitting_degree = gr.Slider(
                        label="fitting degree",
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        value=1,
                    )
                    gr_prompt[task_type] = gr.Textbox(label="shape_guided_prompt")
                    gr_negative_prompt[task_type] = gr.Textbox(label="shape_guided_negative_prompt")
                tab_shape_guided.select(fn=lambda: TASK_LIST[3], inputs=None, outputs=gr_task_radio)

                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=45, step=1)
                    scale = gr.Slider(
                        info="For object removal and image outpainting, it is recommended to set the value at 10 or above.",
                        label="Guidance Scale",
                        minimum=0.1,
                        maximum=30.0,
                        value=7.5,
                        step=0.1,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
            with gr.Column():
                gr.Markdown("### Inpainting result")
                inpaint_result = gr.Gallery(label="Generated images", show_label=False, columns=2)
                gr.Markdown("### Mask")
                gallery = gr.Gallery(label="Generated masks", show_label=False, columns=2)

        # =========================================
        # passing parameters into function call
        # =========================================
        PROMPT_ARGS = list(gr_prompt.values()) + list(gr_negative_prompt.values())
        prefix_args = [
            input_image,
            gr_task_radio,
            fitting_degree,
            vertical_expansion_ratio,
            horizontal_expansion_ratio,
            ddim_steps,
            scale,
            seed,
            input_control_image,
            control_type,
            controlnet_conditioning_scale,
        ]

        def update_click(
            input_image,
            task,
            fitting_degree,
            vertical_expansion_ratio,
            horizontal_expansion_ratio,
            ddim_steps,
            scale,
            seed,
            input_control_image,
            control_type,
            controlnet_conditioning_scale,
            *prompt_args,
        ):
            # parse prompt arguments
            prompt_args = list(prompt_args)
            task_id = TASK_LIST.index(task)
            input_prompt, input_negative_prompt = prompt_args[task_id], prompt_args[task_id + len(TASK_LIST)]

            # parse task prompt
            input_prompt = TASK_PROMPT[args.version][task]["prompt"].format(input_prompt)
            promptA = TASK_PROMPT[args.version][task]["promptA"].format(input_prompt)
            promptB = TASK_PROMPT[args.version][task]["promptB"].format(input_prompt)
            input_negative_prompt = TASK_PROMPT[args.version][task]["negative_prompt"].format(input_negative_prompt)
            negative_promptA = TASK_PROMPT[args.version][task]["negative_promptA"].format(input_negative_prompt)
            negative_promptB = TASK_PROMPT[args.version][task]["negative_promptB"].format(input_negative_prompt)
            if args.version == "ppt1" and task == "text-guided" and input_control_image is not None:
                return controller.predict_controlnet(
                    task,
                    input_prompt,
                    input_negative_prompt,
                    promptA,
                    negative_promptA,
                    promptB,
                    negative_promptB,
                    fitting_degree,
                    input_image,
                    input_control_image,
                    control_type,
                    input_prompt,
                    input_negative_prompt,
                    ddim_steps,
                    scale,
                    seed,
                    controlnet_conditioning_scale,
                )
            else:
                return controller.predict(
                    task,
                    input_prompt,
                    input_negative_prompt,
                    promptA,
                    negative_promptA,
                    promptB,
                    negative_promptB,
                    fitting_degree,
                    input_image,
                    vertical_expansion_ratio,
                    horizontal_expansion_ratio,
                    ddim_steps,
                    scale,
                    seed,
                )

        # set the buttons
        run_button.click(
            fn=update_click,
            inputs=prefix_args + PROMPT_ARGS,
            outputs=[inpaint_result, gallery],
        )

    demo.queue()
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
