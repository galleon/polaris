import random

import numpy as np
import torch
from diffusers import AutoencoderKL, AutoencoderTiny, DiffusionPipeline
from smolagents import Tool

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(device)
good_vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device)
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dtype, vae=taef1).to(device)

torch.cuda.empty_cache()

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048


class PainterTool(Tool):
    name = "painter"
    description = "Generates images"
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The prompt to generate the image from",
        },
    }
    output_type = "image"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "painter"

    def create_images(
        self, prompt, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=3.5, num_inference_steps=28
    ):
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        generator = torch.Generator().manual_seed(seed)
        images = []

        for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            output_type="pil",
            good_vae=good_vae,
        ):
            print(f"Generated image with seed {seed}: {img}")
            images.append(img)

        print(f"Generated image with seed {seed}: {img}")

        return img

    def forward(self, prompt: str) -> str:
        return self.create_images(prompt)
