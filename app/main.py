# app/main.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rembg import remove
from PIL import Image
import io
import os

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Serve form UI
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Upload + Remove background
@app.post("/remove")
async def remove_bg(request: Request, file: UploadFile = File(...)):
    input_bytes = await file.read()
    output_bytes = remove(input_bytes)

    # Save result to disk
    out_path = f"output/{file.filename}.png"
    os.makedirs("output", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(output_bytes)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_url": f"/{out_path}"
    })

# Serve result image
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/static", StaticFiles(directory="static"), name="static")
