import asyncio
import base64
import io
from contextlib import asynccontextmanager

import torch
import trimesh
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="MiniMaker", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    description: str = ""


def _run_image_generation(description: str) -> Image.Image:
    """Load SDXL on demand, generate an image, then unload."""
    from diffusers import StableDiffusionXLPipeline

    prompt = (
        f"{description}, tabletop miniature, plain white background, "
        "studio lighting, full body, centered"
    )

    print(f"[SDXL] Loading pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    print(f"[SDXL] Generating image for: {prompt}")

    result = pipe(prompt=prompt, width=1024, height=1024, num_inference_steps=30)
    image = result.images[0]

    del pipe
    torch.cuda.empty_cache()
    print("[SDXL] Pipeline unloaded, VRAM freed.")

    return image


def _run_shape_generation_from_image(image: Image.Image) -> str:
    """Load Hunyuan3D on demand, generate mesh from image, then unload."""
    import numpy as np
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    print("[Hunyuan3D] Loading pipeline...")
    pipeline_3d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline_3d.to("cuda")
    print("[Hunyuan3D] Generating mesh...")

    mesh_output = pipeline_3d(image=image)

    if hasattr(mesh_output, "mesh") and mesh_output.mesh is not None:
        mesh = mesh_output.mesh
    elif isinstance(mesh_output, list) and len(mesh_output) > 0:
        mesh = mesh_output[0]
    else:
        mesh = mesh_output

    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            mesh = trimesh.Trimesh(
                vertices=np.array(mesh.vertices),
                faces=np.array(mesh.faces),
            )
        else:
            raise ValueError(f"Unexpected mesh output type: {type(mesh)}")

    buffer = io.BytesIO()
    mesh.export(buffer, file_type="stl")
    stl_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    del pipeline_3d
    torch.cuda.empty_cache()
    print("[Hunyuan3D] Pipeline unloaded, VRAM freed.")

    return stl_b64


def _run_full_pipeline(description: str) -> str:
    """Run the full text-to-3D pipeline: SDXL image gen → Hunyuan3D mesh gen."""
    image = _run_image_generation(description)
    stl_b64 = _run_shape_generation_from_image(image)
    return stl_b64


@app.post("/generate-miniature")
async def generate_miniature(req: GenerateRequest):
    loop = asyncio.get_event_loop()
    stl_b64 = await loop.run_in_executor(None, _run_full_pipeline, req.description)
    return {"status": "ok", "model": stl_b64}


# --- Legacy debug endpoints ---

@app.post("/generate")
async def generate(req: GenerateRequest):
    loop = asyncio.get_event_loop()
    image = await loop.run_in_executor(None, _run_image_generation, req.description)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"status": "ok", "image": img_b64}


@app.post("/generate-3d")
async def generate_3d(req: GenerateRequest):
    loop = asyncio.get_event_loop()
    stl_b64 = await loop.run_in_executor(None, _run_shape_generation_from_image,
                                          Image.new("RGB", (1024, 1024), "white"))
    return {"status": "ok", "model": stl_b64}
