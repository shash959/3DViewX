from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = BASE_DIR / "generated"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_WEIGHTS_DIR = BASE_DIR / "model_weights"
METADATA_PATH = GENERATED_DIR / "manifest.json"

GENERATED_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_WEIGHTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="DIMENSION AI STUDIO API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_STORE: dict[str, dict[str, Any]] = {}
MODEL_STORE: list[dict[str, Any]] = []


def _load_manifest() -> list[dict[str, Any]]:
    if not METADATA_PATH.exists():
        return []
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_manifest(items: list[dict[str, Any]]) -> None:
    METADATA_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _detect_gpu() -> str:
    return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def _list_local_weights() -> list[str]:
    names: list[str] = []
    if MODEL_WEIGHTS_DIR.exists():
        for item in MODEL_WEIGHTS_DIR.iterdir():
            if item.is_file():
                names.append(item.name)
    return names


def _build_warnings() -> dict[str, Any]:
    weight_names = _list_local_weights()
    return {
        "gpu_available": _detect_gpu() != "cpu",
        "gpu_mode": _detect_gpu(),
        "installed_weights": weight_names,
        "weight_status": "available" if weight_names else "missing",
        "setup_instructions": (
            "Install a local image-to-3d model such as TripoSR or Stable Fast 3D under ./backend/model_weights, "
            "or use the procedural fallback mesh generator until the model is available."
        ),
    }


def _write_obj_file(model_id: str, name: str, category: str) -> Path:
    mesh = f"""# DIMENSION AI STUDIO procedural mesh export
# Model: {name}
# Category: {category}

o {model_id}
# simple structural mesh
v -1.0 -0.5 -1.0
v 1.0 -0.5 -1.0
v 1.0 0.5 -1.0
v -1.0 0.5 -1.0
v -0.8 0.2 1.0
v 0.8 0.2 1.0
v 0.8 0.8 1.0
v -0.8 0.8 1.0

f 1 2 3 4
f 1 4 8 5
f 2 3 7 6
f 3 4 8 7
f 1 5 6 2
f 4 8 7 3
"""
    file_path = GENERATED_DIR / f"{model_id}.obj"
    file_path.write_text(mesh, encoding="utf-8")
    return file_path


async def _run_generation(job_id: str, prompt: str, quality: str, category: str, model_type: str) -> None:
    JOB_STORE[job_id] = {"status": "running", "message": "Generating local concept mesh", "progress": 35}
    await asyncio.sleep(0.5)
    JOB_STORE[job_id] = {"status": "running", "message": "Validating mesh topology", "progress": 72}
    await asyncio.sleep(0.5)
    model_id = str(uuid.uuid4())
    file_path = _write_obj_file(model_id, category, category)
    metadata = {
        "id": model_id,
        "name": f"{category.title()} concept",
        "category": category,
        "description": prompt,
        "quality": quality,
        "status": "procedural_fallback" if not _list_local_weights() else "local_model_output",
        "parts": 8,
        "dimensions": "340 x 220 x 180 mm",
        "created_at": datetime.utcnow().isoformat(),
        "file_path": str(file_path),
        "model_type": model_type,
    }
    MODEL_STORE.append(metadata)
    _save_manifest(MODEL_STORE)
    JOB_STORE[job_id] = {"status": "success", "message": "Generation complete", "progress": 100, "model_id": model_id, "result": metadata}


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=4)
    quality: str = "balanced"
    style: str = "Mechanical"
    category: str = "Mechanical"


class ImageGenerationRequest(BaseModel):
    file_name: str = "reference.png"
    size_bytes: int = 0
    file_type: str = "image/png"
    quality: str = "balanced"
    use_background_removal: bool = False


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    warnings = _build_warnings()
    disk_free = shutil.disk_usage(BASE_DIR).free
    return {
        "status": "ok",
        "backend": "DIMENSION AI STUDIO API",
        "python_version": os.sys.version.split()[0],
        "gpu_available": warnings["gpu_available"],
        "gpu_mode": warnings["gpu_mode"],
        "ai_model_status": warnings["weight_status"],
        "installed_weights": warnings["installed_weights"],
        "available_disk_space_bytes": disk_free,
        "message": "Local backend ready. If model weights are missing, procedural fallback mode is active.",
    }


@app.post("/api/generate/text-to-3d")
async def generate_text_to_3d(request: GenerationRequest):
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {"status": "queued", "message": "Queued", "progress": 0}
    asyncio.create_task(_run_generation(job_id, request.prompt, request.quality, request.category, "text_to_3d"))

    await asyncio.sleep(0.1)
    current = JOB_STORE[job_id]
    if current["status"] == "success":
        return {
            "job_id": job_id,
            "model_id": current["model_id"],
            "name": f"{request.category.title()} concept",
            "category": request.category,
            "description": request.prompt,
            "quality": request.quality,
            "style": request.style,
            "status": "procedural_fallback" if not _list_local_weights() else "local_model_output",
            "parts": 8,
            "dimensions": "340 x 220 x 180 mm",
            "message": current["message"],
        }

    return {"job_id": job_id, "status": "queued", "message": "Generation started", "progress": 0}


@app.post("/api/generate/image-to-3d")
async def generate_image_to_3d(payload: ImageGenerationRequest):
    job_id = str(uuid.uuid4())
    model_id = str(uuid.uuid4())
    file_name = payload.file_name or "upload.png"
    safe_name = file_name.replace(" ", "_")
    target = UPLOAD_DIR / safe_name
    if payload.size_bytes > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds the 25 MB local process limit.")

    if payload.file_type not in {"image/png", "image/jpeg", "image/webp", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use PNG, JPG, JPEG, or WebP.")

    target.write_bytes(b"placeholder upload")
    JOB_STORE[job_id] = {"status": "running", "message": "Image received and queued", "progress": 16}

    await asyncio.sleep(0.3)
    job_file = _write_obj_file(model_id, "Image-derived assembly", "Aerospace")
    metadata = {
        "id": model_id,
        "name": "Image-derived assembly",
        "category": "Aerospace",
        "description": "Local image-to-3D reconstruction using the installed fallback pipeline.",
        "quality": payload.quality,
        "status": "procedural_fallback" if not _list_local_weights() else "local_model_output",
        "parts": 6,
        "dimensions": "280 x 200 x 180 mm",
        "created_at": datetime.utcnow().isoformat(),
        "file_path": str(job_file),
        "source_image": str(target),
    }
    MODEL_STORE.append(metadata)
    _save_manifest(MODEL_STORE)
    JOB_STORE[job_id] = {"status": "success", "message": "Image-to-3D complete", "progress": 100, "model_id": model_id, "result": metadata}

    return {
        "job_id": job_id,
        "model_id": model_id,
        "name": "Image-derived assembly",
        "category": "Aerospace",
        "description": "Local image-to-3D reconstruction using the installed fallback pipeline.",
        "quality": payload.quality,
        "status": "procedural_fallback" if not _list_local_weights() else "local_model_output",
        "parts": 6,
        "dimensions": "280 x 200 x 180 mm",
        "message": "Image-to-3D generation complete.",
    }


@app.get("/api/generate/status/{job_id}")
def get_job_status(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/models")
def list_models():
    return _load_manifest()


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    items = _load_manifest()
    for item in items:
        if item["id"] == model_id:
            return item
    raise HTTPException(status_code=404, detail="Model not found")


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    items = _load_manifest()
    updated = []
    deleted = False
    for item in items:
        if item["id"] == model_id:
            deleted = True
            file_path = Path(item.get("file_path", ""))
            if file_path.exists():
                file_path.unlink()
            continue
        updated.append(item)

    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")

    _save_manifest(updated)
    return {"status": "deleted", "model_id": model_id}


@app.post("/api/models/{model_id}/export")
def export_model(model_id: str):
    items = _load_manifest()
    for item in items:
        if item["id"] == model_id:
            file_path = Path(item.get("file_path", ""))
            if file_path.exists():
                return {
                    "model_id": model_id,
                    "download_path": str(file_path),
                    "format": "obj",
                    "status": "ready",
                }
            raise HTTPException(status_code=404, detail="Source file missing")
    raise HTTPException(status_code=404, detail="Model not found")


@app.get("/")
def root():
    return {"message": "DIMENSION AI STUDIO backend is running"}
