import random

import cv2
import gradio as gr
import numpy as np
import torch
from controlnet_aux import HEDdetector, OpenposeDetector
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from PIL import Image, ImageFilter
from pipeline.pipeline_PowerPaint import \
    StableDiffusionInpaintPipeline as Pipeline
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from utils.utils import TokenizerWrapper, add_tokens
from model.diffusers_c.models import ImageProjection,UNet2DConditionModel
from transformers import CLIPTextModel
from model.BrushNet_CA import BrushNetModel
from diffusers import UniPCMultistepScheduler
from pipeline.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
import os

base_path = './PowerPaint_v2'
# os.system('apt install git')
# os.system('apt install git-lfs')
# os.system(f'git lfs clone https://code.openxlab.org.cn/zhuangjunhao/PowerPaint_v2.git {base_path}')
# os.system(f'cd {base_path} && git lfs pull')
# os.system(f'cd ..')
torch.set_grad_enabled(False)
context_prompt = ""
context_negative_prompt = ""
base_model_path = "./PowerPaint_v2/realisticVisionV60B1_v51VAE/"
dtype = torch.float16
unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", revision=None,torch_dtype=dtype
    )
text_encoder_brushnet = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", revision=None, torch_dtype=dtype
    )
brushnet = BrushNetModel.from_unet(unet)
global pipe
pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet,text_encoder_brushnet = text_encoder_brushnet, torch_dtype=dtype, low_cpu_mem_usage=False, safety_checker=None
)
pipe.unet = UNet2DConditionModel.from_pretrained(
        base_model_path, subfolder="unet", revision=None,torch_dtype=dtype
    )
pipe.tokenizer = TokenizerWrapper(from_pretrained=base_model_path, subfolder="tokenizer", revision=None)
add_tokens(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder_brushnet,
    placeholder_tokens=['P_ctxt', 'P_shape', 'P_obj'],
    initialize_tokens=['a', 'a', 'a'],
    num_vectors_per_token=10)
from safetensors.torch import load_model
load_model(pipe.brushnet, "./PowerPaint_v2/PowerPaint_Brushnet/diffusion_pytorch_model.safetensors")

pipe.text_encoder_brushnet.load_state_dict(torch.load("./PowerPaint_v2/PowerPaint_Brushnet/pytorch_model.bin"), strict=False)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()
global current_control
current_control = 'canny'
# controlnet_conditioning_scale = 0.8


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def add_task(control_type):
    # print(control_type)
    if control_type == 'object-removal':
        promptA = 'P_ctxt'
        promptB = 'P_ctxt'
        negative_promptA = 'P_obj'
        negative_promptB = 'P_obj'
    elif control_type == 'context-aware':
        promptA = 'P_ctxt'
        promptB = 'P_ctxt'
        negative_promptA = ''
        negative_promptB = ''
    elif control_type == 'shape-guided':
        promptA = 'P_shape'
        promptB = 'P_ctxt'
        negative_promptA = 'P_shape'
        negative_promptB = 'P_ctxt'
    elif control_type == 'image-outpainting':
        promptA = 'P_ctxt'
        promptB = 'P_ctxt'
        negative_promptA = 'P_obj'
        negative_promptB = 'P_obj'
    else:
        promptA = 'P_obj'
        promptB = 'P_obj'
        negative_promptA =  'P_obj'
        negative_promptB =  'P_obj'

    return promptA, promptB, negative_promptA, negative_promptB



def predict(input_image, prompt, fitting_degree, ddim_steps, scale, seed,
            negative_prompt, task,vertical_expansion_ratio,horizontal_expansion_ratio):
    size1, size2 = input_image['image'].convert('RGB').size

    if task!='image-outpainting':
        if size1 < size2:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (640, int(size2 / size1 * 640)))
        else:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (int(size1 / size2 * 640), 640))
    else:
        if size1 < size2:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (512, int(size2 / size1 * 512)))
        else:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (int(size1 / size2 * 512), 512))

    if task=='image-outpainting' or task == 'context-aware':
        prompt = prompt + ' empty scene'
    if task=='object-removal':
        prompt = prompt + ' empty scene blur'
        
    if vertical_expansion_ratio!=None and horizontal_expansion_ratio!=None:
        o_W,o_H = input_image['image'].convert('RGB').size
        c_W = int(horizontal_expansion_ratio*o_W)
        c_H = int(vertical_expansion_ratio*o_H)

        expand_img = np.ones((c_H, c_W,3), dtype=np.uint8)*127
        original_img = np.array(input_image['image'])
        expand_img[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = original_img

        blurry_gap = 10

        expand_mask = np.ones((c_H, c_W,3), dtype=np.uint8)*255
        if vertical_expansion_ratio == 1 and horizontal_expansion_ratio!=1:
            expand_mask[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio!=1:
            expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio==1:
            expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = 0
        
        input_image['image'] = Image.fromarray(expand_img)
        input_image['mask'] = Image.fromarray(expand_mask)

        

    promptA, promptB, negative_promptA, negative_promptB = add_task(task)
    img = np.array(input_image['image'].convert('RGB'))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image['image'] = input_image['image'].resize((H, W))
    input_image['mask'] = input_image['mask'].resize((H, W))

    np_inpimg = np.array(input_image['image'])
    np_inmask = np.array(input_image['mask'])/255.0

    np_inpimg = np_inpimg*(1-np_inmask)

    input_image['image'] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")

    set_seed(seed)

    global pipe
    result = pipe(
            promptA = promptA, 
            promptB = promptB,
            promptU = prompt,
            tradoff=fitting_degree,
            tradoff_nag=fitting_degree,
            image = input_image['image'].convert('RGB'), 
            mask = input_image['mask'].convert('RGB'), 
            num_inference_steps=ddim_steps, 
            generator=torch.Generator("cuda").manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA = negative_promptA,
            negative_promptB = negative_promptB,
            negative_promptU = negative_prompt,
            guidance_scale = scale,
            width=H,
            height=W,
        ).images[0]
    mask_np = np.array(input_image['mask'].convert('RGB'))
    red = np.array(result).astype('float') * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
         mask_np.astype('float') / 512.0 * red).astype('uint8'))
    m_img = input_image['mask'].convert('RGB').filter(
        ImageFilter.GaussianBlur(radius=3))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))

    dict_res = [input_image['mask'].convert('RGB'), result_m]

    dict_out = [result]

    return dict_out, dict_res


def predict2(input_image, prompt, ddim_steps, scale, seed,
            negative_prompt, task,vertical_expansion_ratio,horizontal_expansion_ratio):
    size1, size2 = input_image['image'].convert('RGB').size

    if task!='image-outpainting':
        if size1 < size2:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (640, int(size2 / size1 * 640)))
        else:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (int(size1 / size2 * 640), 640))
    else:
        if size1 < size2:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (512, int(size2 / size1 * 512)))
        else:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (int(size1 / size2 * 512), 512))

    if task=='image-outpainting' or task == 'context-aware':
        prompt = prompt + ' empty scene'
    if task=='object-removal':
        prompt = prompt + ' empty scene blur'
        
    if vertical_expansion_ratio!=None and horizontal_expansion_ratio!=None:
        o_W,o_H = input_image['image'].convert('RGB').size
        c_W = int(horizontal_expansion_ratio*o_W)
        c_H = int(vertical_expansion_ratio*o_H)

        expand_img = np.ones((c_H, c_W,3), dtype=np.uint8)*127
        original_img = np.array(input_image['image'])
        expand_img[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = original_img

        blurry_gap = 10

        expand_mask = np.ones((c_H, c_W,3), dtype=np.uint8)*255
        if vertical_expansion_ratio == 1 and horizontal_expansion_ratio!=1:
            expand_mask[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio!=1:
            expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio==1:
            expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = 0
        
        input_image['image'] = Image.fromarray(expand_img)
        input_image['mask'] = Image.fromarray(expand_mask)

        

    promptA, promptB, negative_promptA, negative_promptB = add_task(task)
    img = np.array(input_image['image'].convert('RGB'))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image['image'] = input_image['image'].resize((H, W))
    input_image['mask'] = input_image['mask'].resize((H, W))
    input_image['ori_image'] = input_image['image'].resize((H, W)).convert("RGB")
    np_inpimg = np.array(input_image['image'])
    np_inmask = np.array(input_image['mask'])/255.0

    np_inpimg = np_inpimg*(1-np_inmask)

    input_image['image'] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")

    set_seed(seed)
    global pipe
    fitting_degree = 1
    result = pipe(
            promptA = promptA, 
            promptB = promptB,
            promptU = prompt,
            tradoff=fitting_degree,
            tradoff_nag=fitting_degree,
            image = input_image['image'].convert('RGB'), 
            mask = input_image['mask'].convert('RGB'), 
            num_inference_steps=ddim_steps, 
            generator=torch.Generator("cuda").manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA = negative_promptA,
            negative_promptB = negative_promptB,
            negative_promptU = negative_prompt,
            guidance_scale = scale,
            width=H,
            height=W,
        ).images[0]
    mask_np = np.array(input_image['mask'].convert('RGB'))
    # mask Image L ==> rgb
    # mask_np [0,255] [h,w,c]
    red = np.array(result).astype('float') * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
         mask_np.astype('float') / 512.0 * red).astype('uint8'))
    m_img = input_image['mask'].convert('RGB').filter(
        ImageFilter.GaussianBlur(radius=3))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))

    dict_res = [input_image['mask'].convert('RGB'), result_m]

    dict_out = [result]

    return dict_out, dict_res

def infer2(input_image, text_guided_prompt, text_guided_negative_prompt,
          ddim_steps, scale, seed, task,removal_prompt,removal_negative_prompt,
          context_prompt,context_negative_prompt):
    if task == 'text-guided':
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt
    # elif task == 'shape-guided':
    #     prompt = shape_guided_prompt
    #     negative_prompt = shape_guided_negative_prompt
    elif task == 'object-removal':
        prompt = removal_prompt
        negative_prompt = removal_negative_prompt
    elif task == 'context-aware':
        prompt = context_prompt
        negative_prompt = context_negative_prompt
    # elif task == 'image-outpainting':
    #     prompt = outpaint_prompt
    #     negative_prompt = outpaint_negative_prompt
    #     return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
    #                    seed, negative_prompt, task,vertical_expansion_ratio,horizontal_expansion_ratio)
    else:
        task = 'text-guided'
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt

    return predict2(input_image, prompt, ddim_steps, scale,
                       seed, negative_prompt, task,None,None)

def infer(input_image, text_guided_prompt, text_guided_negative_prompt,
          shape_guided_prompt, shape_guided_negative_prompt, fitting_degree,
          ddim_steps, scale, seed, task,vertical_expansion_ratio,
          horizontal_expansion_ratio,outpaint_prompt,
          outpaint_negative_prompt,removal_prompt,removal_negative_prompt,
          context_prompt,context_negative_prompt):
    if task == 'text-guided':
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt
    # elif task == 'shape-guided':
    #     prompt = shape_guided_prompt
    #     negative_prompt = shape_guided_negative_prompt
    elif task == 'object-removal':
        prompt = removal_prompt
        negative_prompt = removal_negative_prompt
    elif task == 'context-aware':
        prompt = context_prompt
        negative_prompt = context_negative_prompt
    # elif task == 'image-outpainting':
    #     prompt = outpaint_prompt
    #     negative_prompt = outpaint_negative_prompt
    #     return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
    #                    seed, negative_prompt, task,vertical_expansion_ratio,horizontal_expansion_ratio)
    else:
        task = 'text-guided'
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt

    return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
                       seed, negative_prompt, task,None,None)


def select_tab_text_guided():
    return 'text-guided'


def select_tab_object_removal():
    return 'object-removal'

def select_tab_context_aware():
    return 'context-aware'

def select_tab_image_outpainting():
    return 'image-outpainting'


def select_tab_shape_guided():
    return 'shape-guided'


with gr.Blocks(css='style.css') as demo:
    with gr.Row():
        gr.Markdown(
            "<div align='center'><font size='18'>PowerPaint: High-Quality Versatile Image Inpainting</font></div>"  # noqa
        )
    with gr.Row():
        gr.Markdown(
            "<div align='center'><font size='5'><a href='https://powerpaint.github.io/'>Project Page</a> &ensp;"  # noqa
            "<a href='https://arxiv.org/abs/2312.03594/'>Paper</a> &ensp;"
            "<a href='https://github.com/zhuang2002/PowerPaint'>Code</a> </font></div>"  # noqa
        )
    with gr.Row():
        gr.Markdown(
            "**Note:** Due to network-related factors, the page may experience occasional bugs！ If the inpainting results deviate significantly from expectations, consider toggling between task options to refresh the content."  # noqa
        )
# Attention: Due to network-related factors, the page may experience occasional bugs. If the inpainting results deviate significantly from expectations, consider toggling between task options to refresh the content.
    with gr.Row():
        with gr.Column():
            gr.Markdown('### Input image and draw mask')
            input_image = gr.Image(source='upload', tool='sketch', type='pil')

            task = gr.Radio(['text-guided', 'object-removal'],
                            show_label=False,
                            visible=False)

            # Text-guided object inpainting
            with gr.Tab('Text-guided object inpainting') as tab_text_guided:
                enable_text_guided = gr.Checkbox(
                    label='Enable text-guided object inpainting',
                    value=True,
                    interactive=False)
                text_guided_prompt = gr.Textbox(label='Prompt')
                text_guided_negative_prompt = gr.Textbox(
                    label='negative_prompt')
            tab_text_guided.select(
                fn=select_tab_text_guided, inputs=None, outputs=task)

            # Object removal inpainting
            with gr.Tab('Object removal inpainting') as tab_object_removal:
                enable_object_removal = gr.Checkbox(
                    label='Enable object removal inpainting',
                    value=True,
                    info='The recommended configuration for the Guidance Scale is 10 or higher. \
                    If undesired objects appear in the masked area, \
                    you can address this by specifically increasing the Guidance Scale.',
                    interactive=False)
                removal_prompt = gr.Textbox(label='Prompt')
                removal_negative_prompt = gr.Textbox(
                    label='negative_prompt')
                context_prompt = removal_prompt
                context_negative_prompt = removal_negative_prompt
            tab_object_removal.select(
                fn=select_tab_object_removal, inputs=None, outputs=task)
            
            # # Object image outpainting
            # with gr.Tab('Image outpainting') as tab_image_outpainting:
            #     enable_object_removal = gr.Checkbox(
            #         label='Enable image outpainting',
            #         value=True,
            #         info='The recommended configuration for the Guidance Scale is 15 or higher. \
            #         If unwanted random objects appear in the extended image region, \
            #             you can enhance the cleanliness of the extension area by increasing the Guidance Scale.',
            #         interactive=False)
            #     outpaint_prompt = gr.Textbox(label='Outpainting_prompt')
            #     outpaint_negative_prompt = gr.Textbox(
            #         label='Outpainting_negative_prompt')
            #     horizontal_expansion_ratio = gr.Slider(
            #         label='horizontal expansion ratio',
            #         minimum=1,
            #         maximum=4,
            #         step=0.05,
            #         value=1,
            #     )
            #     vertical_expansion_ratio = gr.Slider(
            #         label='vertical expansion ratio',
            #         minimum=1,
            #         maximum=4,
            #         step=0.05,
            #         value=1,
            #     )
            # tab_image_outpainting.select(
            #     fn=select_tab_image_outpainting, inputs=None, outputs=task)

            # # Shape-guided object inpainting
            # with gr.Tab('Shape-guided object inpainting') as tab_shape_guided:
            #     enable_shape_guided = gr.Checkbox(
            #         label='Enable shape-guided object inpainting',
            #         value=True,
            #         interactive=False)
            #     shape_guided_prompt = gr.Textbox(label='shape_guided_prompt')
            #     shape_guided_negative_prompt = gr.Textbox(
            #         label='shape_guided_negative_prompt')
            #     fitting_degree = gr.Slider(
            #         label='fitting degree',
            #         minimum=0.3,
            #         maximum=1,
            #         step=0.05,
            #         value=1,
            #     )
            # tab_shape_guided.select(
            #     fn=select_tab_shape_guided, inputs=None, outputs=task)

            run_button = gr.Button(label='Run')
            with gr.Accordion('Advanced options', open=False):
                ddim_steps = gr.Slider(
                    label='Steps', minimum=1, maximum=50, value=50, step=1)
                scale = gr.Slider(
                    label='Guidance Scale',
                    minimum=0.1,
                    maximum=45.0,
                    value=12,
                    step=0.1)
                seed = gr.Slider(
                    label='Seed',
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
        with gr.Column():
            gr.Markdown('### Inpainting result')
            inpaint_result = gr.Gallery(
                label='Generated images', show_label=False, columns=2)
            gr.Markdown('### Mask')
            gallery = gr.Gallery(
                label='Generated masks', show_label=False, columns=2)

    shape_guided_negative_prompt = None
    outpaint_prompt = None
    outpaint_negative_prompt = None
    fitting_degree = None
    vertical_expansion_ratio = None
    horizontal_expansion_ratio= None
    shape_guided_prompt = None
    run_button.click(
        fn=infer2,
        inputs=[
            input_image, text_guided_prompt, text_guided_negative_prompt,
            ddim_steps, scale, seed, task,removal_prompt,removal_negative_prompt,
            context_prompt,context_negative_prompt
        ],
        # inputs=[
        #     input_image, text_guided_prompt, text_guided_negative_prompt,
        #     shape_guided_prompt, shape_guided_negative_prompt, fitting_degree,
        #     ddim_steps, scale, seed, task,vertical_expansion_ratio,
        #     horizontal_expansion_ratio,outpaint_prompt,
        #     outpaint_negative_prompt,removal_prompt,removal_negative_prompt,
        #     context_prompt,context_negative_prompt
        # ],
        outputs=[inpaint_result, gallery])

demo.queue()
demo.launch(share=True, server_port=7860)
