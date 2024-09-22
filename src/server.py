#!/usr/bin/env python3

import torch
import argparse
import diffusers
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles as StaticFiles
from pydantic import BaseModel
from typing import Any
import uuid
import os

from diffusers import FluxPipeline
from pick import pick, Option as PickOption

from app_config import AppConfig
from flux_config import FluxConfig
from torch_helper import get_best_device
from flux_server import FluxImageGenerator
from flux_pipeline import FullFluxLoader

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic here
    print("Application startup")
    startup_event()
    yield
    # Shutdown logic here 
    print("Application shutdown")

# Initialize FastAPI app
app = FastAPI(title="FluxP Server API", version="1.0", lifespan=lifespan)

IMAGE_DIR="images"

# Mount the static files directory
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Define the request model
class GenerateRequest(BaseModel):
    prompt: str
    steps: int = None

# Define the response model (adjust as needed based on FluxProompter.run's output)
class GenerateResponse(BaseModel):
    image_url: str

def fix_flux_rope_transformer():
    """
    Fix the transformer used in the flux model to work with MPS.
    Does nothing if the pipeline is not using MPS.
    """

    _flux_rope = diffusers.models.transformers.transformer_flux.rope

    def patched_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0
        if pos.device.type == "mps":
            return _flux_rope(pos.to("cpu"), dim, theta).to(device=pos.device)
        return _flux_rope(pos, dim, theta)

    diffusers.models.transformers.transformer_flux.rope = patched_flux_rope

def load_model_interactive(device: torch.device) -> tuple[FluxPipeline, FluxConfig]:
    model_option = "flux1-schnell"

    # Instantiate model-specific configuration
    config: FluxConfig
    match model_option:
        case "flux1-schnell":
            config = FluxConfig(
                repo="black-forest-labs/FLUX.1-schnell",
                model_display_name="flux1-schnell",
                model_safetensors=None,
                fast_iteration_count=1,
                quality_iteration_count=4,
                fixed_guidance_scale=True,
                default_guidance_scale=0,
                max_sequence_length=256,
                torch_dtype=torch.bfloat16,
            )

    # Load inference pipeline
    pipeline: FluxPipeline = FullFluxLoader(device, config).load()

    # Move pipeline to device
    pipeline.to(device)

    return (pipeline, config)

def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Run FLUX.1 models interactively.")
    parser.add_argument("--offload-cpu", action="store_true", help="Use less VRAM.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU device.")
    args = parser.parse_args()
    return AppConfig(offload_cpu=args.offload_cpu, force_cpu=args.force_cpu)

# Global variables to hold the pipeline and proompter
pipeline = None
model_config = None
proompter = None

#@app.on_event("startup")
def startup_event():
    global pipeline, model_config, proompter

    # Fix transformer for MPS
    fix_flux_rope_transformer()

    # Parse command line arguments
    # Note: When using FastAPI with Uvicorn, command line arguments might need to be handled differently.
    # For simplicity, we'll assume default configuration. Adjust as needed.
    # If you need to pass arguments, consider using environment variables or a configuration file.
    app_config = AppConfig(offload_cpu=False, force_cpu=False)

    # Determine the best device
    device = get_best_device(force_cpu=app_config.force_cpu)

    # Load model config and pipeline
    pipeline, model_config = load_model_interactive(device=device)

    # Offload to CPU if requested
    if app_config.offload_cpu:
        if device.type == "mps":
            raise RuntimeError("Cannot offload to CPU when using MPS.")
        else:
            pipeline.enable_model_cpu_offload()

    proompter = FluxImageGenerator(pipeline, model_config)
    


@app.post("/generate", response_model=GenerateResponse)
def generate_image(request: GenerateRequest, req: Request):
    """
    Generate text based on the provided prompt.
    """
    if not proompter:
        raise HTTPException(status_code=500, detail="Proompter not initialized.")

    try:
        # Validate 'steps' if provided
        steps = request.steps
        if steps is not None:
            if not isinstance(steps, int):
                raise ValueError("'steps' must be an integer.")
            if steps < 1:
                raise ValueError("'steps' must be a positive integer.")

        # Generate a unique filename using UUID
        unique_id = uuid.uuid4().hex
        filename = f"{unique_id}.png"  # Assuming PNG format; adjust as needed

        # Full path to save the image
        image_path = os.path.join(IMAGE_DIR, filename)

        # Run the proompter to generate and save the image
        if steps is not None:
            # Pass 'steps' to the run method
            proompter.run_once(request.prompt, output_path=image_path, steps=steps)
        else:
            # Run without 'steps'
            proompter.run_once(request.prompt, output_path=image_path)

        # Construct the full URL to access the image
        base_url = str(req.base_url).rstrip("/")
        image_url = f"{base_url}/images/{filename}"

        return GenerateResponse(image_url=image_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "OK"}

# Entry point for running with Uvicorn
# To run the server, use the following command:
# uvicorn your_script_name:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    # Parse command line arguments for the server if needed
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
