import os
import re
import time
import torch
from pick import pick
from typing import Iterable
from diffusers import FluxPipeline
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.completion.base import CompleteEvent
from prompt_toolkit.document import Document

from flux_config import FluxConfig

class FluxImageGenerator:
    last_prompt: str
    last_seed: int
    fixed_seed: int | None
    hint_size: tuple[int, int]
    hint_inference_steps: int
    lora_scale: float
    generator: torch.Generator
    pipeline: FluxPipeline
    config: FluxConfig
    session: PromptSession
    completions: Completer

    def __init__(self, pipeline: FluxPipeline, config: FluxConfig):
        self.last_prompt = ""
        self.last_seed = 0
        self.fixed_seed = None
        self.hint_size = (512, 512)
        self.hint_inference_steps = 4
        self.lora_scale = 0.5
        self.generator = torch.Generator("cpu")
        self.pipeline = pipeline
        self.config = config
        self.session = PromptSession()


    def run_once(self, user_prompt: str, output_path:str ,steps: int = 4):

        # Seed shenanigans
        seed = self.generator.seed() if self.fixed_seed is None else self.fixed_seed
        generator = self.generator.manual_seed(seed)
        self.last_seed = seed

        # Party time
        result = self.pipeline(
            user_prompt,
            width=self.hint_size[0],
            height=self.hint_size[1],
            guidance_scale=self.config.default_guidance_scale,
            num_inference_steps=steps,
            max_sequence_length=self.config.max_sequence_length,
            generator=generator,
            joint_attention_kwargs={"scale": self.lora_scale},
        )

        # Create folder for current date
        date = time.strftime("%Y-%m-%d")
        os.makedirs(f"output/{date}", exist_ok=True)

        # Obtain final image and save it
        image = result.images[0]
        image.save(f"output/{date}/{int(time.time())}_{seed}.png")
        image.save("output/latest.png")
        image.save(output_path)

        # Update last prompt
        self.last_prompt = user_prompt

        return image
