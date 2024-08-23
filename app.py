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

import diffusers
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from powerpaint.models import BrushNetModel, UNet2DConditionModel
from powerpaint.pipelines import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPowerPaintBrushNetPipeline,
)


TASK_PROMPTS = {
    "object-removal": {
        "placeholder": "P_ctxt",
    }
}


def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    if control_type == "object-removal" or control_type == "image-outpainting":
        if version == "ppt1":
            pos_prefix = "empty scene blur " + prompt
            neg_prefix = negative_prompt
        promptA = "P_ctxt " + pos_prefix
        promptB = "P_ctxt " + pos_prefix
        negative_promptA = neg_prefix + "P_obj "
        negative_promptB = neg_prefix + "P_obj "
    elif control_type == "shape-guided":
        if version == "ppt1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else:
        if version == "ppt1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


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
                "runwayml/stable-diffusion-inpainting",
                unet=diffusers.UNet2DConditionModel.from_pretrained(self.pretrained_model_path, subfolder="unet").to(
                    "cuda"
                ),
                text_encoder=CLIPTextModel.from_pretrained(self.pretrained_model_path, subfolder="text_encoder").to(
                    "cuda"
                ),
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
        else:
            # brushnet-based version
            self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
                "checkpoints/realisticVisionV60B1_v51VAE",
                unet=UNet2DConditionModel.from_pretrained(
                    "checkpoints/realisticVisionV60B1_v51VAE",
                    subfolder="unet",
                    revision=None,
                    torch_dtype=weight_dtype,
                    local_files_only=local_files_only,
                ),
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
                low_cpu_mem_usage=False,
                safety_checker=None,
            )

        self.pipe.add_tokens(
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initializer_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
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
        input_image,
        task,
        prompt,
        negative_prompt,
        fitting_degree,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        ddim_steps,
        scale,
        seed,
    ):
        image, mask = input_image["image"].convert("RGB"), input_image["mask"].convert("RGB")
        w, h = image.size

        # processing images due to limited memory
        new_size = 640 if task != "image-outpainting" else 512
        image = (
            image.resize((new_size, int(h / w * new_size)))
            if w < h
            else image.resize((int(w / h * new_size), new_size))
        )

        # preparing masks
        if vertical_expansion_ratio is not None and horizontal_expansion_ratio is not None:
            w2, h2 = int(horizontal_expansion_ratio * w), int(vertical_expansion_ratio * h)

            expand_img = Image.new("RGB", (w2, h2), (0, 0, 0))
            expand_img.paste(image, (int((w2 - image.size[0]) / 2), int((h2 - image.size[1]) / 2)))
            expand_img = np.ones((h2, w2, 3), dtype=np.uint8)
            # original_img = np.array(input_image["image"])
            # blury_gap = 10

        if self.version != "ppt1":
            if task == "image-outpainting":
                prompt = prompt + " empty scene"
            if task == "object-removal":
                prompt = prompt + " empty scene blur"
        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print(promptA, promptB, negative_promptA, negative_promptB)

        W, H = int(w - w % 8), int(h - h % 8)
        image = image.resize((H, W))
        mask = mask.resize((H, W))
        set_seed(seed)

        result = self.pipe(
            promptA=promptA,
            promptB=promptB,
            tradeoff=fitting_degree,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            image=image,
            mask=mask,
            width=H,
            height=W,
            guidance_scale=scale,
            num_inference_steps=ddim_steps,
        ).images[0]

        # for brushnet-based method
        np_inpimg = np.array(input_image["image"])
        np_inmask = np.array(input_image["mask"]) / 255.0
        np_inpimg = np_inpimg * (1 - np_inmask)
        input_image["image"] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")
        print(self.pipe.device, self.pipe.unet.device, self.pipe.brushnet.device, self.pipe.text_encoder.device)
        result = self.pipe(
            promptA=promptA,
            promptB=promptB,
            prompt=prompt,
            tradeoff=fitting_degree,
            image=input_image["image"].convert("RGB"),
            mask=input_image["mask"].convert("RGB"),
            num_inference_steps=ddim_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            negative_prompt=negative_prompt,
            guidance_scale=scale,
            width=H,
            height=W,
        ).images[0]

        # highlighted inpainting results
        mask_np = np.array(input_image["mask"].convert("RGB"))
        result_m = np.array(result)
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )

        # paste the inpainting results into original images
        m_img = 255 - np.array(input_image["mask"].convert("RGB"))
        m_img = Image.fromarray(m_img).filter(ImageFilter.GaussianBlur(radius=3))
        m_img = 1.0 - np.asarray(m_img) / 255.0
        m_img = np.asarray(m_img > 0).astype("float")
        original_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
        ours_np = np.asarray(result) / 255.0
        result_paste = ours_np * m_img + (1.0 - m_img) * original_np
        result_paste = Image.fromarray(np.uint8(result_paste * 255))

        # final output
        dict_out = [input_image["image"].convert("RGB"), result_paste]  # [result]
        dict_res = [input_image["mask"].convert("RGB"), result_m]
        return dict_out, dict_res

    def infer(
        self,
        input_image,
        task,
        prompt,
        negative_prompt,
        fitting_degree,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        ddim_steps,
        scale,
        seed,
        input_control_image=None,
        control_type="canny",
        controlnet_conditioning_scale=None,
    ):
        prompt = prompt[task]
        negative_prompt = negative_prompt[task]
        if task == "image-outpainting":
            vertical_expansion_ratio = horizontal_expansion_ratio = None

        # currently, we only support controlnet in PowerPaint-v1
        if self.version == "ppt1" and task == "text-guided" and input_control_image is not None:
            return self.predict_controlnet(
                input_image,
                input_control_image,
                control_type,
                prompt,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                controlnet_conditioning_scale,
            )
        else:
            return self.predict(
                input_image,
                task,
                prompt,
                negative_prompt,
                fitting_degree,
                vertical_expansion_ratio,
                horizontal_expansion_ratio,
                ddim_steps,
                scale,
                seed,
            )


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--pretrained_model_path", type=str, required=True)
    args.add_argument("--base_model_path", type=str, default="")
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
        prompt = {}
        negative_prompt = {}
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input image and draw mask")
                input_image = gr.Image(source="upload", tool="sketch", type="pil")

                task = gr.Radio(
                    ["text-guided", "object-removal", "shape-guided", "image-outpainting"],
                    show_label=False,
                    visible=False,
                )

                # Text-guided object inpainting
                with gr.Tab("Text-guided object inpainting") as tab_text_guided:
                    task_type = "text-guided"
                    enable_text_guided = gr.Checkbox(
                        label="Enable text-guided object inpainting", value=True, interactive=False
                    )
                    prompt[task_type] = gr.Textbox(label="Prompt")
                    negative_prompt[task_type].append(gr.Textbox(label="negative_prompt"))
                    tab_text_guided.select(fn=lambda: task_type, inputs=None, outputs=task)

                    # currently, we only support controlnet in PowerPaint-v1
                    input_control_image = control_type = controlnet_conditioning_scale = None
                    if args.version == "ppt1":
                        gr.Markdown("### Controlnet setting")
                        enable_control = gr.Checkbox(
                            label="Enable controlnet", info="Enable this if you want to use controlnet"
                        )
                        controlnet_conditioning_scale = gr.Slider(
                            label="controlnet conditioning scale",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.5,
                        )
                        control_type = gr.Radio(["canny", "pose", "depth", "hed"], label="Control type")
                        input_control_image = gr.Image(source="upload", type="pil")

                # Object removal inpainting
                with gr.Tab("Object removal inpainting") as tab_object_removal:
                    task_type = "object-removal"
                    enable_object_removal = gr.Checkbox(
                        label="Enable object removal inpainting",
                        value=True,
                        info="The recommended configuration for the Guidance Scale is 10 or higher. \
                        If undesired objects appear in the masked area, \
                        you can address this by specifically increasing the Guidance Scale.",
                        interactive=True,
                    )
                    prompt[task_type].append(gr.Textbox(label="Prompt"))
                    negative_prompt[task_type] = gr.Textbox(label="negative_prompt")
                    tab_object_removal.select(fn=lambda: task_type, inputs=None, outputs=task)

                # image outpainting
                with gr.Tab("Image outpainting") as tab_image_outpainting:
                    task_type = "image-outpainting"
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
                    prompt[task_type].append(gr.Textbox(label="Outpainting_prompt"))
                    negative_prompt[task_type].append(gr.Textbox(label="Outpainting_negative_prompt"))
                    tab_image_outpainting.select(fn=lambda: task_type, inputs=None, outputs=task)

                # Shape-guided object inpainting
                with gr.Tab("Shape-guided object inpainting") as tab_shape_guided:
                    task_type = "shape-guided"
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
                    prompt[task_type].append(gr.Textbox(label="shape_guided_prompt"))
                    negative_prompt[task_type].append(gr.Textbox(label="shape_guided_negative_prompt"))
                    tab_shape_guided.select(fn=lambda: task_type, inputs=None, outputs=task)

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

        run_button.click(
            fn=controller.infer,
            inputs=[
                input_image,
                task,
                prompt,
                negative_prompt,
                fitting_degree,
                vertical_expansion_ratio,
                horizontal_expansion_ratio,
                ddim_steps,
                scale,
                seed,
                input_control_image,
                control_type,
                controlnet_conditioning_scale,
            ],
            outputs=[inpaint_result, gallery],
        )

    demo.queue()
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
