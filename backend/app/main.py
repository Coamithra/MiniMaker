import asyncio
import base64
import functools
import io
from contextlib import asynccontextmanager

import numpy as np
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


# --- Request models ---

class GenerateRequest(BaseModel):
    description: str = ""


class GenerateImagesRequest(BaseModel):
    description: str


class GenerateBackViewRequest(BaseModel):
    image: str  # base64 PNG
    description: str


class GenerateModelRequest(BaseModel):
    front_image: str  # base64 PNG
    back_image: str   # base64 PNG


# --- Helper: encode PIL image to base64 PNG ---

def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# --- Step 6: Mesh thickening for 3D printability ---

def _thicken_mesh(
    mesh: trimesh.Trimesh,
    min_thickness: float | None = None,
) -> trimesh.Trimesh:
    """Thicken thin regions of a mesh by inflating vertices outward along normals.

    Uses a KD-tree to find, for each vertex, the nearest "opposite-facing"
    vertex (one whose normal points back toward it). The distance to that
    vertex approximates local wall thickness. Thin vertices are then pushed
    outward along their normals to meet the minimum thickness.

    Preserves original mesh topology and detail — no voxel round-trip.
    """
    from scipy.spatial import cKDTree

    # Auto-calculate min thickness: 1.5% of the longest bounding box axis.
    # For a 60mm mini this is ~0.9mm — a reasonable 3D-print wall minimum.
    if min_thickness is None:
        min_thickness = max(mesh.bounding_box.extents) * 0.015

    print(f"[Thicken] Min thickness target: {min_thickness:.4f}")
    print(f"[Thicken] Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    vertices = mesh.vertices.copy()
    normals = mesh.vertex_normals.copy()

    # Build KD-tree over all vertices
    print("[Thicken] Building KD-tree and measuring thickness...")
    tree = cKDTree(vertices)

    # For each vertex, query nearby neighbors within a search radius.
    # We only need to search up to min_thickness — anything farther is
    # already thick enough.
    search_radius = min_thickness
    neighbors = tree.query_ball_point(vertices, r=search_radius, workers=-1)

    thickness = np.full(len(vertices), np.inf)
    for i, nbr_indices in enumerate(neighbors):
        if len(nbr_indices) <= 1:
            continue
        nbr_idx = np.array(nbr_indices)
        # Filter to neighbors that are truly "across the wall":
        # 1. The vector to the neighbor must point inward (against our normal)
        # 2. The neighbor's own normal must also point back toward us
        # Both conditions must hold to avoid false positives from surface
        # folds, creases, and adjacent geometry on the same side.
        to_nbr = vertices[nbr_idx] - vertices[i]
        dists = np.linalg.norm(to_nbr, axis=1)
        # Skip self and very close neighbors on the same surface patch
        far_enough = dists > min_thickness * 0.1
        if not far_enough.any():
            continue
        sub_idx = nbr_idx[far_enough]
        to_nbr_unit = to_nbr[far_enough] / dists[far_enough, np.newaxis]
        dists = dists[far_enough]
        # Check 1: neighbor is in inward direction (behind our surface)
        inward_dots = to_nbr_unit @ normals[i]
        # Check 2: neighbor's normal points back toward us (opposing face)
        nbr_normal_dots = np.sum(normals[sub_idx] * normals[i], axis=1)
        across_mask = (inward_dots < -0.3) & (nbr_normal_dots < 0.0)
        if not across_mask.any():
            continue
        thickness[i] = dists[across_mask].min()

    # Find thin vertices and compute how much to push them out
    thin_mask = thickness < min_thickness
    thin_count = thin_mask.sum()
    print(f"[Thicken] Found {thin_count} thin vertices "
          f"({100 * thin_count / len(vertices):.1f}% of mesh)")

    if thin_count == 0:
        print("[Thicken] No thin regions found, returning original mesh.")
        return mesh

    # Push each thin vertex outward by half the deficit (both sides of the
    # wall contribute, so each vertex only needs to move half the difference)
    deficit = min_thickness - thickness[thin_mask]
    displacement = deficit * 0.5
    vertices[thin_mask] += normals[thin_mask] * displacement[:, np.newaxis]

    thickened = trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy())

    print(f"[Thicken] Done. Displaced {thin_count} vertices, "
          f"max displacement: {displacement.max():.4f}")
    return thickened


# --- Step 5 helpers ---

def _run_batch_image_generation(description: str) -> list[str]:
    """Load SDXL, generate 4 images in 2 batches of 2, return base64 PNGs."""
    from diffusers import StableDiffusionXLPipeline

    prompt = (
        f"{description}, tabletop miniature, no background, "
        "studio lighting, full body, centered, front view"
    )

    print("[SDXL] Loading pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    print(f"[SDXL] Generating 4 images for: {prompt}")

    images: list[str] = []
    for batch_idx in range(2):
        try:
            result = pipe(
                prompt=prompt,
                width=1024,
                height=1024,
                num_inference_steps=30,
                num_images_per_prompt=2,
            )
            images.extend(_pil_to_b64(img) for img in result.images)
        except torch.cuda.OutOfMemoryError:
            print(f"[SDXL] OOM on batch {batch_idx}, falling back to sequential")
            torch.cuda.empty_cache()
            for _ in range(2):
                result = pipe(
                    prompt=prompt,
                    width=1024,
                    height=1024,
                    num_inference_steps=30,
                    num_images_per_prompt=1,
                )
                images.append(_pil_to_b64(result.images[0]))

    del pipe
    torch.cuda.empty_cache()
    print(f"[SDXL] Done, generated {len(images)} images. Pipeline unloaded.")
    return images


def _run_back_view_generation(front_b64: str, description: str) -> str:
    """Load SDXL img2img, generate back view of front image, return base64 PNG."""
    from diffusers import StableDiffusionXLImg2ImgPipeline

    front_image = _b64_to_pil(front_b64).resize((1024, 1024))

    prompt = (
        f"{description}, same character, view from behind, back view, "
        "tabletop miniature, no background, studio lighting, full body, centered"
    )

    print("[SDXL img2img] Loading pipeline...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    print(f"[SDXL img2img] Generating back view...")

    result = pipe(
        prompt=prompt,
        image=front_image,
        strength=0.7,
        num_inference_steps=30,
    )
    back_b64 = _pil_to_b64(result.images[0])

    del pipe
    torch.cuda.empty_cache()
    print("[SDXL img2img] Pipeline unloaded, VRAM freed.")
    return back_b64


def _run_mv_shape_generation(front_b64: str, back_b64: str) -> str:
    """Load Hunyuan3D-2mv, generate mesh from front+back views, return base64 STL."""
    import numpy as np
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    front_image = _b64_to_pil(front_b64)
    back_image = _b64_to_pil(back_b64)

    print("[Hunyuan3D-2mv] Loading pipeline...")
    pipeline_3d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline_3d.to("cuda")
    print("[Hunyuan3D-2mv] Generating mesh from front+back views...")

    mesh_output = pipeline_3d(image={"front": front_image, "back": back_image})

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

    del pipeline_3d
    torch.cuda.empty_cache()
    print("[Hunyuan3D-2mv] Pipeline unloaded, VRAM freed.")

    mesh = _thicken_mesh(mesh)

    buffer = io.BytesIO()
    mesh.export(buffer, file_type="stl")
    stl_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return stl_b64


# --- Step 5 endpoints ---

@app.post("/generate-images")
async def generate_images(req: GenerateImagesRequest):
    loop = asyncio.get_event_loop()
    images = await loop.run_in_executor(None, _run_batch_image_generation, req.description)
    return {"status": "ok", "images": images}


@app.post("/generate-back-view")
async def generate_back_view(req: GenerateBackViewRequest):
    loop = asyncio.get_event_loop()
    fn = functools.partial(_run_back_view_generation, req.image, req.description)
    image = await loop.run_in_executor(None, fn)
    return {"status": "ok", "image": image}


@app.post("/generate-model")
async def generate_model(req: GenerateModelRequest):
    loop = asyncio.get_event_loop()
    fn = functools.partial(_run_mv_shape_generation, req.front_image, req.back_image)
    stl_b64 = await loop.run_in_executor(None, fn)
    return {"status": "ok", "model": stl_b64}


# --- Legacy endpoints (kept for debugging) ---

def _run_image_generation(description: str) -> Image.Image:
    """Load SDXL on demand, generate an image, then unload."""
    from diffusers import StableDiffusionXLPipeline

    prompt = (
        f"{description}, tabletop miniature, no background, "
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

    del pipeline_3d
    torch.cuda.empty_cache()
    print("[Hunyuan3D] Pipeline unloaded, VRAM freed.")

    mesh = _thicken_mesh(mesh)

    buffer = io.BytesIO()
    mesh.export(buffer, file_type="stl")
    stl_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

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
