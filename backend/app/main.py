import base64
import io
from contextlib import asynccontextmanager

import torch
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading SDXL pipeline (this may download ~6.5GB on first run)...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipeline.to("cuda")
    print("SDXL pipeline loaded and ready.")
    yield
    del pipeline
    torch.cuda.empty_cache()


app = FastAPI(title="MiniMaker", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT = (
    "a highly detailed D&D tabletop miniature figure, painted, "
    "heroic fantasy character, dramatic pose, studio lighting, "
    "white background, product photography, 8k"
)


class GenerateRequest(BaseModel):
    description: str = ""


@app.post("/generate")
async def generate(req: GenerateRequest):
    image = pipeline(
        prompt=PROMPT,
        num_inference_steps=30,
        width=1024,
        height=1024,
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"status": "ok", "image": img_base64}
