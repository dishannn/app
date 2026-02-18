from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from rembg import new_session, remove

app = FastAPI(title="Passport Photo Generator")

# Better human segmentation model
session = new_session("u2net_human_seg")


@dataclass
class PassportJob:
    passport_photo: Image.Image
    bg_color: str
    photos_per_row: int = 5
    rows: int = 2
    meta: Dict[str, str] = field(default_factory=dict)


jobs: Dict[str, PassportJob] = {}


class PhotoSettingsUpdate(BaseModel):
    bg_color: str | None = Field(default=None, pattern=r"^#(?:[0-9a-fA-F]{3}){1,2}$")
    photos_per_row: int | None = Field(default=None, ge=1, le=10)
    rows: int | None = Field(default=None, ge=1, le=20)


class JobCreatedResponse(BaseModel):
    job_id: str
    preview_url: str
    download_url: str


def _create_passport_photo(input_image: Image.Image, bg_color: str) -> Image.Image:
    output_image = remove(input_image.convert("RGBA"), session=session).convert("RGBA")

    background = Image.new("RGBA", output_image.size, bg_color)
    final_img = Image.alpha_composite(background, output_image)

    width, height = final_img.size
    crop_width = int(width * 0.8)
    crop_height = int(crop_width * 45 / 35)

    left = (width - crop_width) // 2
    top = int(height * 0.1)

    cropped = final_img.crop((left, top, left + crop_width, top + crop_height))

    passport = cropped.resize((413, 531), Image.LANCZOS)

    passport_np = np.array(passport)
    passport_np = np.clip(passport_np * 1.05, 0, 255).astype(np.uint8)
    return Image.fromarray(passport_np)


def _build_sheet(passport: Image.Image, photos_per_row: int, rows: int) -> Image.Image:
    border_size = 8
    bordered_passport = Image.new(
        "RGB",
        (passport.width + border_size * 2, passport.height + border_size * 2),
        "black",
    )
    bordered_passport.paste(passport.convert("RGB"), (border_size, border_size))

    photo_width, photo_height = bordered_passport.size

    sheet_width = 2480
    sheet_height = 3508
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")

    top_margin = 100
    side_margin = 100

    x_spacing = (
        sheet_width - 2 * side_margin - (photos_per_row * photo_width)
    ) // max(1, photos_per_row - 1)
    y_spacing = 80

    for r in range(rows):
        for c in range(photos_per_row):
            x = side_margin + c * (photo_width + x_spacing)
            y = top_margin + r * (photo_height + y_spacing)
            if y + photo_height <= sheet_height:
                sheet.paste(bordered_passport, (x, y))

    return sheet


def _get_job(job_id: str) -> PassportJob:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/passport/process", response_model=JobCreatedResponse)
async def process_passport_photo(
    file: UploadFile = File(...),
    bg_color: str = Form("#ffffff"),
    photos_per_row: int = Form(5),
    rows: int = Form(2),
) -> JobCreatedResponse:
    contents = await file.read()

    try:
        input_image = Image.open(io.BytesIO(contents))
    except Exception as exc:  # Pillow raises different subclasses depending on file
        raise HTTPException(status_code=400, detail="invalid image file") from exc

    passport_photo = _create_passport_photo(input_image, bg_color)

    job_id = str(uuid.uuid4())
    jobs[job_id] = PassportJob(
        passport_photo=passport_photo,
        bg_color=bg_color,
        photos_per_row=photos_per_row,
        rows=rows,
        meta={"filename": file.filename or "uploaded-image"},
    )

    return JobCreatedResponse(
        job_id=job_id,
        preview_url=f"/passport/{job_id}/preview",
        download_url=f"/passport/{job_id}/download?format=pdf",
    )


@app.patch("/passport/{job_id}/settings")
def update_photo_settings(job_id: str, payload: PhotoSettingsUpdate) -> dict:
    job = _get_job(job_id)

    if payload.bg_color is not None:
        job.bg_color = payload.bg_color
    if payload.photos_per_row is not None:
        job.photos_per_row = payload.photos_per_row
    if payload.rows is not None:
        job.rows = payload.rows

    return {
        "message": "settings updated",
        "job_id": job_id,
        "photos_per_row": job.photos_per_row,
        "rows": job.rows,
        "bg_color": job.bg_color,
    }


@app.get("/passport/{job_id}/preview")
def preview_passport_sheet(
    job_id: str,
    photos_per_row: int | None = None,
    rows: int | None = None,
) -> StreamingResponse:
    job = _get_job(job_id)
    sheet = _build_sheet(
        job.passport_photo,
        photos_per_row=photos_per_row or job.photos_per_row,
        rows=rows or job.rows,
    )

    img_io = io.BytesIO()
    sheet.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(
        img_io,
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=preview_{job_id}.png"},
    )


@app.get("/passport/{job_id}/download")
def download_passport_sheet(
    job_id: str,
    photos_per_row: int | None = None,
    rows: int | None = None,
    format: str = "pdf",
) -> StreamingResponse:
    job = _get_job(job_id)
    sheet = _build_sheet(
        job.passport_photo,
        photos_per_row=photos_per_row or job.photos_per_row,
        rows=rows or job.rows,
    )

    file_io = io.BytesIO()
    if format.lower() == "pdf":
        sheet.save(file_io, format="PDF", resolution=300)
        media_type = "application/pdf"
        filename = "passport_sheet.pdf"
    elif format.lower() == "png":
        sheet.save(file_io, format="PNG")
        media_type = "image/png"
        filename = "passport_sheet.png"
    else:
        raise HTTPException(status_code=400, detail="format must be pdf or png")

    file_io.seek(0)
    return StreamingResponse(
        file_io,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
