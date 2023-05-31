from fastapi import FastAPI, File
from fastapi.responses import FileResponse
import torch
from diffusers import DPMSolverMultistepScheduler
from optimum.graphcore.diffusers import get_default_ipu_configs, INFERENCE_ENGINES_TO_MODEL_NAMES, IPUStableDiffusionPipeline
import os


app = FastAPI()

pod_type = os.getenv("GRAPHCORE_POD_TYPE", "pod16")
executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/stablediffusion2_text2img"

engine = "stable-diffusion-768-v2-0"  # maps to "stabilityai/stable-diffusion-2"
model_name = INFERENCE_ENGINES_TO_MODEL_NAMES[engine]
image_width = os.getenv("STABLE_DIFFUSION_2_TXT2IMG_DEFAULT_WIDTH", default=768)
image_height = os.getenv("STABLE_DIFFUSION_2_TXT2IMG_DEFAULT_HEIGHT", default=768)


unet_ipu_config, text_encoder_ipu_config, vae_ipu_config, safety_checker_ipu_config = \
get_default_ipu_configs(
    engine=engine, width=image_width, height=image_height, pod_type=pod_type, 
    executable_cache_dir=executable_cache_dir 
)

pipe = IPUStableDiffusionPipeline.from_pretrained(
    model_name,
    revision="fp16", 
    torch_dtype=torch.float16,
    requires_safety_checker=False,
    unet_ipu_config=unet_ipu_config,
    text_encoder_ipu_config=text_encoder_ipu_config,
    vae_ipu_config=vae_ipu_config,
    safety_checker_ipu_config=safety_checker_ipu_config
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Sample run to generate the compiled graph
pipe("apple", height=image_height, width=image_width, num_inference_steps=25, guidance_scale=9)


@app.get("/")
async def root():
    return {"message": "Hello World!"}

@app.post("/generate")
async def predict(prompt: str):

    image = pipe(prompt, height=image_height, width=image_width, num_inference_steps=25, guidance_scale=9).images[0]
    image.save("generated_image.jpeg")

    return FileResponse('generated_image.jpeg', media_type="image/jpeg")