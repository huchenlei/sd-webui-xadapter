import os
from dataclasses import dataclass
from copy import copy
from collections import defaultdict
from typing import List, Callable

import gradio as gr
import torch

from scripts.lib_xadapter.logging import logger
from scripts.lib_xadapter.adapter import Adapter_XL
from modules import scripts, sd_models, sd_samplers, paths
from modules.processing import StableDiffusionProcessing, process_images, Processed

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


class ResidualModifier(torch.nn.Module):
    def __init__(
        self,
        original_layer: torch.nn.Module,
        extra_residuals: Callable[[], torch.Tensor],
    ) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.extra_residuals = extra_residuals

    def forward(self, *args, **kwargs):
        return self.original_layer(*args, **kwargs) + self.extra_residuals()


class AdapterRunner:
    def __init__(
        self,
        hidden_states: List[List[torch.Tensor]],
        weight: float = 1.0,
    ) -> None:
        self.hidden_states = (h for h in hidden_states)
        self.weight = weight
        self.adapter = Script.adapter
        assert self.adapter is not None

        # Output of X-Adapter, which should be add back to main pass up block
        # outputs.
        self.adapter_output = []

    def run(self) -> torch.Tensor:
        if not self.adapter_output:
            self.adapter_output = self.adapter(next(self.hidden_states))
        return self.adapter_output.pop(0) * self.weight


def hook_up_layers(
    output_blocks: List[torch.nn.Module],
    sd15_hidden_states: List[List[torch.Tensor]],
    weight: float = 1.0,
) -> List[torch.nn.Module]:
    assert Script.adapter is not None
    runner = AdapterRunner(sd15_hidden_states, weight=weight)
    return [
        ResidualModifier(block, extra_residuals=runner.run) for block in output_blocks
    ]


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

    @staticmethod
    def run_sd15(p: StableDiffusionProcessing, adapter_args: XAdapterArgs) -> Processed:
        """Run SD1.5 generation to get hidden states."""
        p2 = copy(p)
        # Mark the process as xadapter pass, so that we do not fall into infinite loop.
        setattr(p2, IS_XADAPTER_PASS_FLAG, True)
        p2.width = adapter_args.width
        p2.height = adapter_args.height
        p2.sampler = adapter_args.sampler
        p2.override_settings = copy(p.override_settings)
        p2.override_settings["sd_model_checkpoint"] = adapter_args.checkpoint

        try:
            processed = process_images(p2)
        finally:
            p2.close()
            # Reload main pass sd model after we are done with sd15 pass.
            sd_models.reload_model_weights()
        return processed

    def process(self, p: StableDiffusionProcessing, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        # Key is up-block num
        # Inner list indices are denoising timesteps.
        self.hidden_states = defaultdict(list)

        adapter_args = XAdapterArgs(*args)
        if not adapter_args.enabled:
            return
        sd_ldm = p.sd_model
        unet = sd_ldm.model.diffusion_model

        if getattr(p, IS_XADAPTER_PASS_FLAG, False):
            # SD15 pass.
            # Collect hidden_states of last 3 upsample blocks.
            for i, layer in enumerate(
                [
                    unet.output_blocks[7],
                    unet.output_blocks[10],
                    unet.out,
                ]
            ):
                hook_module_input(layer, i, self.hidden_states)
            logger.info("Preprocess pass output blocks hooked.")
        else:
            Script.run_sd15(p, adapter_args)
            # SDXL pass.
            # Apply X-Adapter
            if Script.adapter is None:
                Script.adapter = load_xadapter(MODEL_FILE)
                Script.adapter.to(device="cuda")
            # Convert defaultdict to matrix.
            hidden_statess = [
                self.hidden_states[i] for i in range(len(self.hidden_states))
            ]
            sd_ldm = p.sd_model
            unet = sd_ldm.model.diffusion_model
            unet.middle_block, unet.output_blocks[2], unet.output_blocks[5] = (
                hook_up_layers(
                    # Expected layer output shapes.
                    # (1280, 32, 32), (1280, 64, 64), (640, 128, 128)
                    [unet.middle_block, unet.output_blocks[2], unet.output_blocks[5]],
                    sd15_hidden_states=list(
                        zip(*hidden_statess)
                    ),  # Transpose the matrix.
                    weight=adapter_args.weight,
                )
            )
            logger.info("Main pass output blocks hooked.")

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """
        adapter_args = XAdapterArgs(*args)
        if not adapter_args.enabled:
            return
        if not getattr(p, IS_XADAPTER_PASS_FLAG, False):
            assert hasattr(self, "hidden_states")
            self.hidden_states.clear()
