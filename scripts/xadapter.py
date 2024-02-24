import os
from dataclasses import dataclass
from copy import copy
from collections import defaultdict
from typing import List

import gradio as gr
import torch

from scripts.logging import logger
from scripts.lib_xadapter.adapter import Adapter_XL
from modules import scripts, sd_models, sd_samplers, paths
from modules.processing import StableDiffusionProcessing, process_images

IS_XADAPTER_PASS_FLAG = "is_xadapter_preprocess_pass"
MODELS_DIR = os.path.join(paths.models_path, "xadapter")
os.makedirs(MODELS_DIR, exist_ok=True)
# Currently there is only 1 model available, so just hardcode the model path.
MODEL_FILE = os.path.join(MODELS_DIR, "X_Adapter_v1.bin")


@dataclass
class XAdapterArgs:
    enabled: bool = False
    # The first pass checkpoint name.
    checkpoint: str = "None"
    sampler: str = "Default"
    width: int = 512
    height: int = 512

    start: float = 0.0
    weight: float = 1.0


def load_xadapter(checkpoint_path: str) -> Adapter_XL:
    ckpt = torch.load(checkpoint_path)
    adapter = Adapter_XL()
    adapter.load_state_dict(ckpt)
    return adapter


# The set of modules that has XAdapter hooks. Avoid hooking module multiple times.
hooked_modules = set()


def hook_module_input(module: torch.nn.Module, i: int, out_dict: dict):
    def hook(module, input, output):
        """Stores the hidden states of a given timestamp for later use.
        Note: 'input' is a tuple of all input arguments to the forward method.

        We hijacks the input of output_block instead of taking the output
        of output_block as implementation of ControlNet patches unet forward
        and does extra work to compute final input to the next output block.
        """
        out_dict[i].append(input[0])

    if module in hooked_modules:
        return
    hooked_modules.add(module)
    module.register_forward_hook(hook)


def get_hidden_states(
    p: StableDiffusionProcessing, adapter_args: XAdapterArgs
) -> List[List[torch.Tensor]]:
    """Run SD1.5 generation to get hidden states."""
    # Do SD1.5 pass
    p2 = copy(p)
    # Mark the process as xadapter pass, so that we do not fall into infinite loop.
    setattr(p2, IS_XADAPTER_PASS_FLAG, True)
    p2.width = adapter_args.width
    p2.height = adapter_args.height
    p2.sampler = adapter_args.sampler
    p2.override_settings = copy(p.override_settings)
    p2.override_settings["sd_model_checkpoint"] = adapter_args.checkpoint

    # Key is up-block num
    # Inner list indices are denoising timesteps.
    hidden_states = defaultdict(list)
    # Note: skip the first output_block as its input is the output of
    # middle block.
    sd_ldm = p2.sd_model
    unet = sd_ldm.model.diffusion_model
    for i, block in enumerate(unet.output_blocks[1:] + [unet.out]):
        hook_module_input(block, i, hidden_states)
    try:
        process_images(p2)
    finally:
        p2.close()
        hidden_states.clear()
    return [hidden_states[i] for i in range(len(hidden_states))]


class UpBlocksModifier(torch.nn.Module):
    def __init__(
        self, original_block: torch.nn.Module, extra_residuals: List[torch.Tensor]
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.extra_residuals = extra_residuals

    def forward(self, *args, **kwargs):
        original_output = self.original_block(*args, **kwargs)
        if self.extra_residuals:
            original_output += self.extra_residuals.pop()
        else:
            logger.warn("Empty residuals detected.")
        return original_output


class Script(scripts.Script):
    adapter = None

    def title(self):
        return "X-Adapter"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        with gr.Row():
            gr.HTML(
                '<a href="https://github.com/showlab/X-Adapter">&nbsp X-Adapter</a><br>'
            )
        with gr.Row():
            enabled = gr.Checkbox(
                label="Enable",
                value=False,
            )
        with gr.Row():
            model = gr.Dropdown(
                label="Adapter model",
                choices=["None"] + sd_models.checkpoint_tiles(),
                value="None",
            )
            sampler = gr.Dropdown(
                label="Adapter sampler",
                choices=[s.name for s in sd_samplers.samplers],
                value="Default",
            )
        with gr.Row():
            width = gr.Slider(
                label="Adapter width", minimum=64, maximum=2048, step=8, value=512
            )
            height = gr.Slider(
                label="Adapter height", minimum=64, maximum=2048, step=8, value=512
            )
        with gr.Row():
            start = gr.Slider(
                label="Adapter start", minimum=0.0, maximum=1.0, step=0.01, value=0.5
            )
            scale = gr.Slider(
                label="Adapter scale", minimum=0.0, maximum=1.0, step=0.01, value=1.0
            )
        return enabled, model, sampler, width, height, start, scale

    def process(self, p: StableDiffusionProcessing, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()

        """
        adapter_args = XAdapterArgs(*args)
        if not adapter_args.enabled or getattr(p, IS_XADAPTER_PASS_FLAG, False):
            return

        # Apply X-Adapter
        hidden_states_by_timesteps = []
        if Script.adapter is None:
            Script.adapter = load_xadapter(MODEL_FILE)
        # Transpose the matrix.
        for hidden_states in zip(*get_hidden_states(p, adapter_args)):
            # hidden states tensor * num of blocks
            hidden_states_by_timesteps.append(Script.adapter(hidden_states))
        # Transpose back.
        hidden_states_by_blocks = list(zip(*hidden_states_by_timesteps))

        sd_ldm = p.sd_model
        unet = sd_ldm.model.diffusion_model
        new_output_blocks = torch.nn.ModuleList()
        for hidden_states, block in zip(
            unet.output_blocks,
            hidden_states_by_blocks,
        ):
            new_output_blocks = UpBlocksModifier(
                block, extra_residuals=hidden_states[::-1]
            )
        unet.output_blocks = new_output_blocks
