# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import fnmatch
import importlib
import inspect
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import requests
import torch
from huggingface_hub import (
    ModelCard,
    create_repo,
    hf_hub_download,
    model_info,
    snapshot_download,
)
from huggingface_hub.utils import OfflineModeIsEnabled, validate_hf_hub_args
from packaging import version
from requests.exceptions import HTTPError
from tqdm.auto import tqdm

from .. import __version__
from ..configuration_utils import ConfigMixin
from ..models import AutoencoderKL
from ..models.attention_processor import FusedAttnProcessor2_0
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from ..schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from ..utils import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    BaseOutput,
    PushToHubMixin,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    is_torch_npu_available,
    is_torch_version,
    logging,
    numpy_to_pil,
)
from ..utils.hub_utils import load_or_create_model_card, populate_model_card
from ..utils.torch_utils import is_compiled_module


if is_torch_npu_available():
    import torch_npu  # noqa: F401


from .pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    CONNECTED_PIPES_KEYS,
    CUSTOM_PIPELINE_FILE_NAME,
    LOADABLE_CLASSES,
    _fetch_class_library_tuple,
    _get_pipeline_class,
    _unwrap_model,
    is_safetensors_compatible,
    load_sub_model,
    maybe_raise_or_warn,
    variant_compatible_siblings,
    warn_deprecated_model_variant,
)


if is_accelerate_available():
    import accelerate


LIBRARIES = []
for library in LOADABLE_CLASSES:
    LIBRARIES.append(library)

logger = logging.get_logger(__name__)


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class AudioPipelineOutput(BaseOutput):
    """
    Output class for audio pipelines.

    Args:
        audios (`np.ndarray`)
            List of denoised audio samples of a NumPy array of shape `(batch_size, num_channels, sample_rate)`.
    """

    audios: np.ndarray


class DiffusionPipeline(ConfigMixin, PushToHubMixin):
    r"""
    Base class for all pipelines.

    [`DiffusionPipeline`] stores all components (models, schedulers, and processors) for diffusion pipelines and
    provides methods for loading, downloading and saving models. It also includes methods to:

        - move all PyTorch modules to the device of your choice
        - enable/disable the progress bar for the denoising iteration

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.
        - **_optional_components** (`List[str]`) -- List of all optional components that don't have to be passed to the
          pipeline to function (should be overridden by subclasses).
    """

    config_name = "model_index.json"
    model_cpu_offload_seq = None
    _optional_components = []
    _exclude_from_cpu_offload = []
    _load_connected_pipes = False
    _is_onnx = False

    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def __setattr__(self, name: str, value: Any):
        if name in self.__dict__ and hasattr(self.config, name):
            # We need to overwrite the config if name exists in config
            if isinstance(getattr(self.config, name), (tuple, list)):
                if value is not None and self.config[name][0] is not None:
                    class_library_tuple = _fetch_class_library_tuple(value)
                else:
                    class_library_tuple = (None, None)

                self.register_to_config(**{name: class_library_tuple})
            else:
                self.register_to_config(**{name: value})

        super().__setattr__(name, value)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
        class implements both a save and loading method. The pipeline is easily reloaded using the
        [`~DiffusionPipeline.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a pipeline to. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name", None)
        model_index_dict.pop("_diffusers_version", None)
        model_index_dict.pop("_module", None)
        model_index_dict.pop("_name_or_path", None)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", False)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}
        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            # Dynamo wraps the original model in a private class.
            # I didn't find a public API to get the original class.
            if is_compiled_module(sub_model):
                sub_model = _unwrap_model(sub_model)
                model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                if library_name in sys.modules:
                    library = importlib.import_module(library_name)
                else:
                    logger.info(
                        f"{library_name} is not installed. Cannot save {pipeline_component_name} as {library_classes} from {library_name}"
                    )

                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            if save_method_name is None:
                logger.warn(f"self.{pipeline_component_name}={sub_model} of type {type(sub_model)} cannot be saved.")
                # make sure that unsaveable components are not tried to be loaded afterward
                self.register_to_config(**{pipeline_component_name: (None, None)})
                continue

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

        # finally save the config
        self.save_config(save_directory)

        if push_to_hub:
            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(repo_id, token=token, is_pipeline=True)
            model_card = populate_model_card(model_card)
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    def to(self, *args, **kwargs):
        r"""
        Performs Pipeline dtype and/or device conversion. A torch.dtype and torch.device are inferred from the
        arguments of `self.to(*args, **kwargs).`

        <Tip>

            If the pipeline already has the correct torch.dtype and torch.device, then it is returned as is. Otherwise,
            the returned pipeline is a copy of self with the desired torch.dtype and torch.device.

        </Tip>


        Here are the ways to call `to`:

        - `to(dtype, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
        - `to(device, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
        - `to(device=None, dtype=None, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the
          specified [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) and
          [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)

        Arguments:
            dtype (`torch.dtype`, *optional*):
                Returns a pipeline with the specified
                [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
            device (`torch.Device`, *optional*):
                Returns a pipeline with the specified
                [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
            silence_dtype_warnings (`str`, *optional*, defaults to `False`):
                Whether to omit warnings if the target `dtype` is not compatible with the target `device`.

        Returns:
            [`DiffusionPipeline`]: The pipeline converted to specified `dtype` and/or `dtype`.
        """
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        silence_dtype_warnings = kwargs.pop("silence_dtype_warnings", False)

        dtype_arg = None
        device_arg = None
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_arg = args[0]
            else:
                device_arg = torch.device(args[0]) if args[0] is not None else None
        elif len(args) == 2:
            if isinstance(args[0], torch.dtype):
                raise ValueError(
                    "When passing two arguments, make sure the first corresponds to `device` and the second to `dtype`."
                )
            device_arg = torch.device(args[0]) if args[0] is not None else None
            dtype_arg = args[1]
        elif len(args) > 2:
            raise ValueError("Please make sure to pass at most two arguments (`device` and `dtype`) `.to(...)`")

        if dtype is not None and dtype_arg is not None:
            raise ValueError(
                "You have passed `dtype` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        dtype = dtype or dtype_arg

        if device is not None and device_arg is not None:
            raise ValueError(
                "You have passed `device` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        device = device or device_arg

        # throw warning if pipeline is in "offloaded"-mode but user tries to manually set to GPU.
        def module_is_sequentially_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.14.0"):
                return False

            return hasattr(module, "_hf_hook") and not isinstance(
                module._hf_hook, (accelerate.hooks.CpuOffload, accelerate.hooks.AlignDevicesHook)
            )

        def module_is_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.17.0.dev0"):
                return False

            return hasattr(module, "_hf_hook") and isinstance(module._hf_hook, accelerate.hooks.CpuOffload)

        # .to("cuda") would raise an error if the pipeline is sequentially offloaded, so we raise our own to make it clearer
        pipeline_is_sequentially_offloaded = any(
            module_is_sequentially_offloaded(module) for _, module in self.components.items()
        )
        if pipeline_is_sequentially_offloaded and device and torch.device(device).type == "cuda":
            raise ValueError(
                "It seems like you have activated sequential model offloading by calling `enable_sequential_cpu_offload`, but are now attempting to move the pipeline to GPU. This is not compatible with offloading. Please, move your pipeline `.to('cpu')` or consider removing the move altogether if you use sequential offloading."
            )

        # Display a warning in this case (the operation succeeds but the benefits are lost)
        pipeline_is_offloaded = any(module_is_offloaded(module) for _, module in self.components.items())
        if pipeline_is_offloaded and device and torch.device(device).type == "cuda":
            logger.warning(
                f"It seems like you have activated model offloading by calling `enable_model_cpu_offload`, but are now manually moving the pipeline to GPU. It is strongly recommended against doing so as memory gains from offloading are likely to be lost. Offloading automatically takes care of moving the individual components {', '.join(self.components.keys())} to GPU when needed. To make sure offloading works as expected, you should consider moving the pipeline back to CPU: `pipeline.to('cpu')` or removing the move altogether if you use offloading."
            )

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        is_offloaded = pipeline_is_offloaded or pipeline_is_sequentially_offloaded
        for module in modules:
            is_loaded_in_8bit = hasattr(module, "is_loaded_in_8bit") and module.is_loaded_in_8bit

            if is_loaded_in_8bit and dtype is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in 8bit and conversion to {dtype} is not yet supported. Module is still in 8bit precision."
                )

            if is_loaded_in_8bit and device is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in 8bit and moving it to {dtype} via `.to()` is not yet supported. Module is still on {module.device}."
                )
            else:
                module.to(device, dtype)

            if (
                module.dtype == torch.float16
                and str(device) in ["cpu"]
                and not silence_dtype_warnings
                and not is_offloaded
            ):
                logger.warning(
                    "Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device. It"
                    " is not recommended to move them to `cpu` as running them will fail. Please make"
                    " sure to use an accelerator to run the pipeline in inference, due to the lack of"
                    " support for`float16` operations on this device in PyTorch. Please, remove the"
                    " `torch_dtype=torch.float16` argument, or use another device for inference."
                )
        return self

    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.device

        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Returns:
            `torch.dtype`: The torch dtype on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.dtype

        return torch.float32

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                      pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                      the custom pipeline.
                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current main branch of GitHub.
                    - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. Defaults to the latest stable 🤗 Diffusers version.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `None`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import DiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

        >>> # Use a different scheduler
        >>> from diffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.scheduler = scheduler
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            if pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            cached_folder = cls.download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                from_flax=from_flax,
                use_safetensors=use_safetensors,
                use_onnx=use_onnx,
                custom_pipeline=custom_pipeline,
                custom_revision=custom_revision,
                variant=variant,
                load_connected_pipeline=load_connected_pipeline,
                **kwargs,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        config_dict = cls.load_config(cached_folder)

        # pop out "_ignore_files" as it is only needed for download
        config_dict.pop("_ignore_files", None)

        # 2. Define which model components should load variants
        # We retrieve the information by matching whether variant
        # model checkpoints exist in the subfolders
        model_variants = {}
        if variant is not None:
            for folder in os.listdir(cached_folder):
                folder_path = os.path.join(cached_folder, folder)
                is_folder = os.path.isdir(folder_path) and folder in config_dict
                variant_exists = is_folder and any(
                    p.split(".")[1].startswith(variant) for p in os.listdir(folder_path)
                )
                if variant_exists:
                    model_variants[folder] = variant

        # 3. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        custom_class_name = None
        if os.path.isfile(os.path.join(cached_folder, f"{custom_pipeline}.py")):
            custom_pipeline = os.path.join(cached_folder, f"{custom_pipeline}.py")
        elif isinstance(config_dict["_class_name"], (list, tuple)) and os.path.isfile(
            os.path.join(cached_folder, f"{config_dict['_class_name'][0]}.py")
        ):
            custom_pipeline = os.path.join(cached_folder, f"{config_dict['_class_name'][0]}.py")
            custom_class_name = config_dict["_class_name"][1]

        pipeline_class = _get_pipeline_class(
            cls,
            config_dict,
            load_connected_pipeline=load_connected_pipeline,
            custom_pipeline=custom_pipeline,
            class_name=custom_class_name,
            cache_dir=cache_dir,
            revision=custom_revision,
        )

        # DEPRECATED: To be removed in 1.0.0
        if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
            version.parse(config_dict["_diffusers_version"]).base_version
        ) <= version.parse("0.5.1"):
            from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy

            pipeline_class = StableDiffusionInpaintPipelineLegacy

            deprecation_message = (
                "You are using a legacy checkpoint for inpainting with Stable Diffusion, therefore we are loading the"
                f" {StableDiffusionInpaintPipelineLegacy} class instead of {StableDiffusionInpaintPipeline}. For"
                " better inpainting results, we strongly suggest using Stable Diffusion's official inpainting"
                " checkpoint: https://huggingface.co/runwayml/stable-diffusion-inpainting instead or adapting your"
                f" checkpoint {pretrained_model_name_or_path} to the format of"
                " https://huggingface.co/runwayml/stable-diffusion-inpainting. Note that we do not actively maintain"
                " the {StableDiffusionInpaintPipelineLegacy} class and will likely remove it in version 1.0.0."
            )
            deprecate("StableDiffusionInpaintPipelineLegacy", "1.0.0", deprecation_message, standard_warn=False)

        # 4. Define expected modules given pipeline signature
        # and define non-None initialized modules (=`init_kwargs`)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs and make sure that optional component modules are filtered out
        init_kwargs = {
            k: init_dict.pop(k)
            for k in optional_kwargs
            if k in init_dict and k not in pipeline_class._optional_components
        }
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # Special case: safety_checker must be loaded separately when using `from_flax`
        if from_flax and "safety_checker" in init_dict and "safety_checker" not in passed_class_obj:
            raise NotImplementedError(
                "The safety checker cannot be automatically loaded when loading weights `from_flax`."
                " Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker"
                " separately if you need it."
            )

        # 5. Throw nice warnings / errors for fast accelerate loading
        if len(unused_kwargs) > 0:
            logger.warning(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )

        # import it here to avoid circular import
        from diffusers import pipelines

        # 6. Load each module in the pipeline
        for name, (library_name, class_name) in logging.tqdm(init_dict.items(), desc="Loading pipeline components..."):
            # 6.1 - now that JAX/Flax is an official framework of the library, we might load from Flax names
            class_name = class_name[4:] if class_name.startswith("Flax") else class_name

            # 6.2 Define all importable classes
            is_pipeline_module = hasattr(pipelines, library_name)
            importable_classes = ALL_IMPORTABLE_CLASSES
            loaded_sub_model = None

            # 6.3 Use passed sub model or load class_name from library_name
            if name in passed_class_obj:
                # if the model is in a pipeline module, then we load it from the pipeline
                # check that passed_class_obj has correct parent class
                maybe_raise_or_warn(
                    library_name, library, class_name, importable_classes, passed_class_obj, name, is_pipeline_module
                )

                loaded_sub_model = passed_class_obj[name]
            else:
                # load sub model
                loaded_sub_model = load_sub_model(
                    library_name=library_name,
                    class_name=class_name,
                    importable_classes=importable_classes,
                    pipelines=pipelines,
                    is_pipeline_module=is_pipeline_module,
                    pipeline_class=pipeline_class,
                    torch_dtype=torch_dtype,
                    provider=provider,
                    sess_options=sess_options,
                    device_map=device_map,
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    offload_state_dict=offload_state_dict,
                    model_variants=model_variants,
                    name=name,
                    from_flax=from_flax,
                    variant=variant,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    cached_folder=cached_folder,
                )
                logger.info(
                    f"Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}."
                )

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        if pipeline_class._load_connected_pipes and os.path.isfile(os.path.join(cached_folder, "README.md")):
            modelcard = ModelCard.load(os.path.join(cached_folder, "README.md"))
            connected_pipes = {prefix: getattr(modelcard.data, prefix, [None])[0] for prefix in CONNECTED_PIPES_KEYS}
            load_kwargs = {
                "cache_dir": cache_dir,
                "resume_download": resume_download,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "token": token,
                "revision": revision,
                "torch_dtype": torch_dtype,
                "custom_pipeline": custom_pipeline,
                "custom_revision": custom_revision,
                "provider": provider,
                "sess_options": sess_options,
                "device_map": device_map,
                "max_memory": max_memory,
                "offload_folder": offload_folder,
                "offload_state_dict": offload_state_dict,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "variant": variant,
                "use_safetensors": use_safetensors,
            }

            def get_connected_passed_kwargs(prefix):
                connected_passed_class_obj = {
                    k.replace(f"{prefix}_", ""): w for k, w in passed_class_obj.items() if k.split("_")[0] == prefix
                }
                connected_passed_pipe_kwargs = {
                    k.replace(f"{prefix}_", ""): w for k, w in passed_pipe_kwargs.items() if k.split("_")[0] == prefix
                }

                connected_passed_kwargs = {**connected_passed_class_obj, **connected_passed_pipe_kwargs}
                return connected_passed_kwargs

            connected_pipes = {
                prefix: DiffusionPipeline.from_pretrained(
                    repo_id, **load_kwargs.copy(), **get_connected_passed_kwargs(prefix)
                )
                for prefix, repo_id in connected_pipes.items()
                if repo_id is not None
            }

            for prefix, connected_pipe in connected_pipes.items():
                # add connected pipes to `init_kwargs` with <prefix>_<component_name>, e.g. "prior_text_encoder"
                init_kwargs.update(
                    {"_".join([prefix, name]): component for name, component in connected_pipe.components.items()}
                )

        # 7. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 8. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        # 9. Save where the model was instantiated from
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        return model

    @property
    def name_or_path(self) -> str:
        return getattr(self.config, "_name_or_path", None)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        # CPU offload models that are not in the seq chain unless they are explicitly excluded
        # these models will stay on CPU until maybe_free_model_hooks is called
        # some models cannot be in the seq chain because they are iteratively called, such as controlnet
        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        for hook in self._all_hooks:
            # offload model and remove hook from model
            hook.offload()
            hook.remove()

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload(device=getattr(self, "_offload_device", "cuda"))

    def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using 🤗 Accelerate, significantly reducing memory usage. When called, the state
        dicts of all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are saved to CPU
        and then moved to `torch.device('meta')` and loaded to GPU only when their specific submodule has its `forward`
        method called. Offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                # make sure to offload buffers if not all high level weights
                # are of type nn.Module
                offload_buffers = len(model._parameters) > 0
                cpu_offload(model, device, offload_buffers=offload_buffers)

    @classmethod
    @validate_hf_hub_args
    def download(cls, pretrained_model_name, **kwargs) -> Union[str, os.PathLike]:
        r"""
        Download and cache a PyTorch diffusion pipeline from pretrained pipeline weights.

        Parameters:
            pretrained_model_name (`str` or `os.PathLike`, *optional*):
                A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                hosted on the Hub.
            custom_pipeline (`str`, *optional*):
                Can be either:

                    - A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained
                      pipeline hosted on the Hub. The repository must contain a file called `pipeline.py` that defines
                      the custom pipeline.

                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current `main` branch of GitHub.

                    - A path to a *directory* (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                For more information on how to load and create custom pipelines, take a look at [How to contribute a
                community pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/contribute_pipeline).

            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `False`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom pipelines and components defined on the Hub in their own files. This
                option should only be set to `True` for repositories you trust and in which you have read the code, as
                it will execute code present on the Hub on your local machine.

        Returns:
            `os.PathLike`:
                A path to the downloaded pipeline.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`.

        </Tip>

        """
        cache_dir = kwargs.pop("cache_dir", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)
        trust_remote_code = kwargs.pop("trust_remote_code", False)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        allow_patterns = None
        ignore_patterns = None

        model_info_call_error: Optional[Exception] = None
        if not local_files_only:
            try:
                info = model_info(pretrained_model_name, token=token, revision=revision)
            except (HTTPError, OfflineModeIsEnabled, requests.ConnectionError) as e:
                logger.warn(f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache.")
                local_files_only = True
                model_info_call_error = e  # save error to reraise it if model is not cached locally

        if not local_files_only:
            config_file = hf_hub_download(
                pretrained_model_name,
                cls.config_name,
                cache_dir=cache_dir,
                revision=revision,
                proxies=proxies,
                force_download=force_download,
                resume_download=resume_download,
                token=token,
            )

            config_dict = cls._dict_from_json_file(config_file)
            ignore_filenames = config_dict.pop("_ignore_files", [])

            # retrieve all folder_names that contain relevant files
            folder_names = [k for k, v in config_dict.items() if isinstance(v, list) and k != "_class_name"]

            filenames = {sibling.rfilename for sibling in info.siblings}
            model_filenames, variant_filenames = variant_compatible_siblings(filenames, variant=variant)

            diffusers_module = importlib.import_module(__name__.split(".")[0])
            pipelines = getattr(diffusers_module, "pipelines")

            # optionally create a custom component <> custom file mapping
            custom_components = {}
            for component in folder_names:
                module_candidate = config_dict[component][0]

                if module_candidate is None or not isinstance(module_candidate, str):
                    continue

                # We compute candidate file path on the Hub. Do not use `os.path.join`.
                candidate_file = f"{component}/{module_candidate}.py"

                if candidate_file in filenames:
                    custom_components[component] = module_candidate
                elif module_candidate not in LOADABLE_CLASSES and not hasattr(pipelines, module_candidate):
                    raise ValueError(
                        f"{candidate_file} as defined in `model_index.json` does not exist in {pretrained_model_name} and is not a module in 'diffusers/pipelines'."
                    )

            if len(variant_filenames) == 0 and variant is not None:
                deprecation_message = (
                    f"You are trying to load the model files of the `variant={variant}`, but no such modeling files are available."
                    f"The default model files: {model_filenames} will be loaded instead. Make sure to not load from `variant={variant}`"
                    "if such variant modeling files are not available. Doing so will lead to an error in v0.24.0 as defaulting to non-variant"
                    "modeling files is deprecated."
                )
                deprecate("no variant default", "0.24.0", deprecation_message, standard_warn=False)

            # remove ignored filenames
            model_filenames = set(model_filenames) - set(ignore_filenames)
            variant_filenames = set(variant_filenames) - set(ignore_filenames)

            # if the whole pipeline is cached we don't have to ping the Hub
            if revision in DEPRECATED_REVISION_ARGS and version.parse(
                version.parse(__version__).base_version
            ) >= version.parse("0.22.0"):
                warn_deprecated_model_variant(pretrained_model_name, token, variant, revision, model_filenames)

            model_folder_names = {os.path.split(f)[0] for f in model_filenames if os.path.split(f)[0] in folder_names}

            custom_class_name = None
            if custom_pipeline is None and isinstance(config_dict["_class_name"], (list, tuple)):
                custom_pipeline = config_dict["_class_name"][0]
                custom_class_name = config_dict["_class_name"][1]

            # all filenames compatible with variant will be added
            allow_patterns = list(model_filenames)

            # allow all patterns from non-model folders
            # this enables downloading schedulers, tokenizers, ...
            allow_patterns += [f"{k}/*" for k in folder_names if k not in model_folder_names]
            # add custom component files
            allow_patterns += [f"{k}/{f}.py" for k, f in custom_components.items()]
            # add custom pipeline file
            allow_patterns += [f"{custom_pipeline}.py"] if f"{custom_pipeline}.py" in filenames else []
            # also allow downloading config.json files with the model
            allow_patterns += [os.path.join(k, "config.json") for k in model_folder_names]

            allow_patterns += [
                SCHEDULER_CONFIG_NAME,
                CONFIG_NAME,
                cls.config_name,
                CUSTOM_PIPELINE_FILE_NAME,
            ]

            load_pipe_from_hub = custom_pipeline is not None and f"{custom_pipeline}.py" in filenames
            load_components_from_hub = len(custom_components) > 0

            if load_pipe_from_hub and not trust_remote_code:
                raise ValueError(
                    f"The repository for {pretrained_model_name} contains custom code in {custom_pipeline}.py which must be executed to correctly "
                    f"load the model. You can inspect the repository content at https://hf.co/{pretrained_model_name}/blob/main/{custom_pipeline}.py.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )

            if load_components_from_hub and not trust_remote_code:
                raise ValueError(
                    f"The repository for {pretrained_model_name} contains custom code in {'.py, '.join([os.path.join(k, v) for k,v in custom_components.items()])} which must be executed to correctly "
                    f"load the model. You can inspect the repository content at {', '.join([f'https://hf.co/{pretrained_model_name}/{k}/{v}.py' for k,v in custom_components.items()])}.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )

            # retrieve passed components that should not be downloaded
            pipeline_class = _get_pipeline_class(
                cls,
                config_dict,
                load_connected_pipeline=load_connected_pipeline,
                custom_pipeline=custom_pipeline,
                repo_id=pretrained_model_name if load_pipe_from_hub else None,
                hub_revision=revision,
                class_name=custom_class_name,
                cache_dir=cache_dir,
                revision=custom_revision,
            )
            expected_components, _ = cls._get_signature_keys(pipeline_class)
            passed_components = [k for k in expected_components if k in kwargs]

            if (
                use_safetensors
                and not allow_pickle
                and not is_safetensors_compatible(
                    model_filenames, variant=variant, passed_components=passed_components
                )
            ):
                raise EnvironmentError(
                    f"Could not find the necessary `safetensors` weights in {model_filenames} (variant={variant})"
                )
            if from_flax:
                ignore_patterns = ["*.bin", "*.safetensors", "*.onnx", "*.pb"]
            elif use_safetensors and is_safetensors_compatible(
                model_filenames, variant=variant, passed_components=passed_components
            ):
                ignore_patterns = ["*.bin", "*.msgpack"]

                use_onnx = use_onnx if use_onnx is not None else pipeline_class._is_onnx
                if not use_onnx:
                    ignore_patterns += ["*.onnx", "*.pb"]

                safetensors_variant_filenames = {f for f in variant_filenames if f.endswith(".safetensors")}
                safetensors_model_filenames = {f for f in model_filenames if f.endswith(".safetensors")}
                if (
                    len(safetensors_variant_filenames) > 0
                    and safetensors_model_filenames != safetensors_variant_filenames
                ):
                    logger.warn(
                        f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(safetensors_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(safetensors_model_filenames - safetensors_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                    )
            else:
                ignore_patterns = ["*.safetensors", "*.msgpack"]

                use_onnx = use_onnx if use_onnx is not None else pipeline_class._is_onnx
                if not use_onnx:
                    ignore_patterns += ["*.onnx", "*.pb"]

                bin_variant_filenames = {f for f in variant_filenames if f.endswith(".bin")}
                bin_model_filenames = {f for f in model_filenames if f.endswith(".bin")}
                if len(bin_variant_filenames) > 0 and bin_model_filenames != bin_variant_filenames:
                    logger.warn(
                        f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(bin_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(bin_model_filenames - bin_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                    )

            # Don't download any objects that are passed
            allow_patterns = [
                p for p in allow_patterns if not (len(p.split("/")) == 2 and p.split("/")[0] in passed_components)
            ]

            if pipeline_class._load_connected_pipes:
                allow_patterns.append("README.md")

            # Don't download index files of forbidden patterns either
            ignore_patterns = ignore_patterns + [f"{i}.index.*json" for i in ignore_patterns]

            re_ignore_pattern = [re.compile(fnmatch.translate(p)) for p in ignore_patterns]
            re_allow_pattern = [re.compile(fnmatch.translate(p)) for p in allow_patterns]

            expected_files = [f for f in filenames if not any(p.match(f) for p in re_ignore_pattern)]
            expected_files = [f for f in expected_files if any(p.match(f) for p in re_allow_pattern)]

            snapshot_folder = Path(config_file).parent
            pipeline_is_cached = all((snapshot_folder / f).is_file() for f in expected_files)

            if pipeline_is_cached and not force_download:
                # if the pipeline is cached, we can directly return it
                # else call snapshot_download
                return snapshot_folder

        user_agent = {"pipeline_class": cls.__name__}
        if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
            user_agent["custom_pipeline"] = custom_pipeline

        # download all allow_patterns - ignore_patterns
        try:
            cached_folder = snapshot_download(
                pretrained_model_name,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )

            # retrieve pipeline class from local file
            cls_name = cls.load_config(os.path.join(cached_folder, "model_index.json")).get("_class_name", None)
            cls_name = cls_name[4:] if isinstance(cls_name, str) and cls_name.startswith("Flax") else cls_name

            diffusers_module = importlib.import_module(__name__.split(".")[0])
            pipeline_class = getattr(diffusers_module, cls_name, None) if isinstance(cls_name, str) else None

            if pipeline_class is not None and pipeline_class._load_connected_pipes:
                modelcard = ModelCard.load(os.path.join(cached_folder, "README.md"))
                connected_pipes = sum([getattr(modelcard.data, k, []) for k in CONNECTED_PIPES_KEYS], [])
                for connected_pipe_repo_id in connected_pipes:
                    download_kwargs = {
                        "cache_dir": cache_dir,
                        "resume_download": resume_download,
                        "force_download": force_download,
                        "proxies": proxies,
                        "local_files_only": local_files_only,
                        "token": token,
                        "variant": variant,
                        "use_safetensors": use_safetensors,
                    }
                    DiffusionPipeline.download(connected_pipe_repo_id, **download_kwargs)

            return cached_folder

        except FileNotFoundError:
            # Means we tried to load pipeline with `local_files_only=True` but the files have not been found in local cache.
            # This can happen in two cases:
            # 1. If the user passed `local_files_only=True`                    => we raise the error directly
            # 2. If we forced `local_files_only=True` when `model_info` failed => we raise the initial error
            if model_info_call_error is None:
                # 1. user passed `local_files_only=True`
                raise
            else:
                # 2. we forced `local_files_only=True` when `model_info` failed
                raise EnvironmentError(
                    f"Cannot load model {pretrained_model_name}: model is not cached locally and an error occurred"
                    " while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace"
                    " above."
                ) from model_info_call_error

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        optional_names = list(optional_parameters)
        for name in optional_names:
            if name in cls._optional_components:
                expected_modules.add(name)
                optional_parameters.remove(name)

        return expected_modules, optional_parameters

    @property
    def components(self) -> Dict[str, Any]:
        r"""
        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations without reallocating additional memory.

        Returns (`dict`):
            A dictionary containing all the modules needed to initialize the pipeline.

        Examples:

        ```py
        >>> from diffusers import (
        ...     StableDiffusionPipeline,
        ...     StableDiffusionImg2ImgPipeline,
        ...     StableDiffusionInpaintPipeline,
        ... )

        >>> text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        >>> inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        ```
        """
        expected_modules, optional_parameters = self._get_signature_keys(self)
        components = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }

        if set(components.keys()) != expected_modules:
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected_modules} to be defined, but {components.keys()} are defined."
            )

        return components

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a NumPy image or a batch of images to a PIL image.
        """
        return numpy_to_pil(images)

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/). When this
        option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
        up during training is not guaranteed.

        <Tip warning={true}>

        ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
        precedent.

        </Tip>

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        >>> # Workaround for not accepting attention shape using VAE for Flash Attention
        >>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            fn_recursive_set_mem_eff(module)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation. When this option is enabled, the attention module splits the input tensor
        in slices to compute attention in several steps. For more than one attention head, the computation is performed
        sequentially over each head. This is useful to save some memory in exchange for a small speed decrease.

        <Tip warning={true}>

        ⚠️ Don't enable attention slicing if you're already using `scaled_dot_product_attention` (SDPA) from PyTorch
        2.0 or xFormers. These attention computations are already very memory efficient so you won't need to enable
        this function. If you enable attention slicing with SDPA or xFormers, it can lead to serious slow downs!

        </Tip>

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5",
        ...     torch_dtype=torch.float16,
        ...     use_safetensors=True,
        ... )

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> pipe.enable_attention_slicing()
        >>> image = pipe(prompt).images[0]
        ```
        """
        self.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously called, attention is
        computed in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def set_attention_slice(self, slice_size: Optional[int]):
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module) and hasattr(m, "set_attention_slice")]

        for module in modules:
            module.set_attention_slice(slice_size)


class StableDiffusionMixin:
    r"""
    Helper for DiffusionPipeline with vae and unet.(mainly for LDM such as stable diffusion)
    """

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    def fuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
        self.fusing_unet = False
        self.fusing_vae = False

        if unet:
            self.fusing_unet = True
            self.unet.fuse_qkv_projections()
            self.unet.set_attn_processor(FusedAttnProcessor2_0())

        if vae:
            if not isinstance(self.vae, AutoencoderKL):
                raise ValueError("`fuse_qkv_projections()` is only supported for the VAE of type `AutoencoderKL`.")

            self.fusing_vae = True
            self.vae.fuse_qkv_projections()
            self.vae.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """Disable QKV projection fusion if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.

        """
        if unet:
            if not self.fusing_unet:
                logger.warning("The UNet was not initially fused for QKV projections. Doing nothing.")
            else:
                self.unet.unfuse_qkv_projections()
                self.fusing_unet = False

        if vae:
            if not self.fusing_vae:
                logger.warning("The VAE was not initially fused for QKV projections. Doing nothing.")
            else:
                self.vae.unfuse_qkv_projections()
                self.fusing_vae = False
