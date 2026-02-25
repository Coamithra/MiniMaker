import asyncio
import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import trimesh
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

pipeline_3d = None

ASSETS_DIR = Path(__file__).parent / "assets"
PRESET_IMAGE = ASSETS_DIR / "preset.png"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_3d
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    print("Loading Hunyuan3D-2mini pipeline (downloads ~3-5GB on first run)...")
    pipeline_3d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline_3d.to("cuda")
    print("Hunyuan3D-2mini pipeline loaded and ready.")
    yield
    del pipeline_3d
    torch.cuda.empty_cache()


app = FastAPI(title="MiniMaker", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    description: str = ""


@app.post("/generate")
async def generate(req: GenerateRequest):
    return {"status": "error", "message": "SDXL not loaded in Step 3. Use /generate-3d instead."}


def _run_shape_generation():
    """Run Hunyuan3D shape generation synchronously (called via executor)."""
    image = Image.open(PRESET_IMAGE).convert("RGB")

    mesh_output = pipeline_3d(image=image)

    if hasattr(mesh_output, "mesh") and mesh_output.mesh is not None:
        mesh = mesh_output.mesh
    elif isinstance(mesh_output, list) and len(mesh_output) > 0:
        mesh = mesh_output[0]
    else:
        mesh = mesh_output

    # Convert to trimesh if it's not already
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            import numpy as np
            mesh = trimesh.Trimesh(
                vertices=np.array(mesh.vertices),
                faces=np.array(mesh.faces),
            )
        else:
            raise ValueError(f"Unexpected mesh output type: {type(mesh)}")

    buffer = io.BytesIO()
    mesh.export(buffer, file_type="stl")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.post("/generate-3d")
async def generate_3d():
    loop = asyncio.get_event_loop()
    glb_base64 = await loop.run_in_executor(None, _run_shape_generation)
    return {"status": "ok", "model": glb_base64}
