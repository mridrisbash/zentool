from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rembg import remove
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import io, os
import cv2
import numpy as np
from fastapi.responses import PlainTextResponse

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# ========= ROUTE: Background Remover =========

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

@app.post("/remove")
async def remove_bg(request: Request, file: UploadFile = File(...)):
    input_bytes = await file.read()
    output_bytes = remove(input_bytes)

    # Save as transparent PNG
    out_path = f"output/{file.filename}.png"
    with open(out_path, "wb") as f:
        f.write(output_bytes)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_url": f"/output/{file.filename}.png"
    })


@app.get("/sitemap.xml", response_class=HTMLResponse)
async def sitemap():
    return FileResponse("static/sitemap.xml", media_type="application/xml")

@app.get("/ads.txt", response_class=PlainTextResponse)
async def ads_txt():
    with open("ads.txt", "r") as f:
        content = f.read()
# ========= ROUTE: Product Photo Enhancer =========

@app.get("/product", response_class=HTMLResponse)
async def product_page(request: Request):
    return templates.TemplateResponse("product.html", {"request": request})

@app.post("/product")
async def enhance_product_photo(
    request: Request,
    file: UploadFile = File(...),
    bg: str = Form("white"),
    shadow: bool = Form(False)
):
    input_bytes = await file.read()
    removed = remove(input_bytes)

    image = Image.open(io.BytesIO(removed)).convert("RGBA")
    image = image.resize((800, 800))

    # Background setup
    if bg == "white":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    elif bg == "branded":
        background = Image.new("RGBA", image.size, (240, 248, 255, 255))
    else:
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))

    # Shadow effect
    if shadow:
        shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        offset = (10, 10)
        shadow_blur = 12
        alpha = image.split()[3].point(lambda p: p > 0 and 80)
        black = Image.new("RGBA", image.size, (0, 0, 0, 255))
        shadow_layer.paste(black, offset, mask=alpha)
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
        background = Image.alpha_composite(background, shadow_layer)

    background.paste(image, (0, 0), image)
    final = background.convert("RGB")

    filename = f"{file.filename}_product.jpg"
    out_path = f"output/{filename}"
    final.save(out_path, format="JPEG", quality=85)

    return templates.TemplateResponse("product.html", {
        "request": request,
        "result_url": f"/output/{filename}"
    })

@app.get("/pfp", response_class=HTMLResponse)
async def pfp_form(request: Request):
    return templates.TemplateResponse("pfp.html", {"request": request})

@app.post("/pfp")
async def pfp_generator(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("default")
):
    try:
        input_bytes = await file.read()
        img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")

        # Crop to square
        size = min(img.width, img.height)
        left = (img.width - size) // 2
        top = (img.height - size) // 2
        img = img.crop((left, top, left + size, top + size))
        img = img.resize((512, 512))

        # Apply style (optional, expand later)
        if style == "cartoon":
            img = img.filter(ImageEnhance.Sharpness(img).enhance(2.0))
        elif style == "anime":
            img = ImageEnhance.Color(img).enhance(2.0)
        elif style == "business":
            img = ImageEnhance.Contrast(img).enhance(1.3)

        # Apply circular mask
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, img.size[0], img.size[1]), fill=255)
        img.putalpha(mask)

        bg = Image.new("RGBA", img.size, (255, 255, 255, 0))
        bg.paste(img, mask=img)

        final = bg.convert("RGB")
        filename = f"{file.filename}_pfp.jpg"
        path = f"output/{filename}"
        final.save(path, format="JPEG", quality=95)

        return templates.TemplateResponse("pfp.html", {
            "request": request,
            "result_url": f"/output/{filename}"
        })

    except Exception as e:
        return templates.TemplateResponse("pfp.html", {
            "request": request,
            "error": f"Error: {e}"
        })

# ========= ROUTE: Passport/ID Photo Generator =========

@app.get("/passport", response_class=HTMLResponse)
async def passport_form(request: Request):
    return templates.TemplateResponse("passport.html", {"request": request})

@app.post("/passport")
async def passport_generator(
    request: Request,
    file: UploadFile = File(...),
    country: str = Form("US"),
    bg_color: str = Form("white")
):
    input_bytes = await file.read()

    try:
        no_bg = remove(input_bytes)
        img_array = np.frombuffer(no_bg, np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        face_cascade = cv2.CascadeClassifier("app/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return templates.TemplateResponse("passport.html", {"request": request, "error": "No face found"})

        x, y, w, h = faces[0]
        cx, cy = x + w // 2, y + h // 2

        size_map = {
            "US": (600, 600),
            "UK": (413, 531),
            "Nigeria": (413, 531)
        }
        out_size = size_map.get(country, (600, 600))

        pad_x = int(w * 1.2)
        pad_y = int(h * 1.6)
        h_img, w_img = img_cv.shape[:2]
        x1 = max(cx - pad_x, 0)
        x2 = min(cx + pad_x, w_img)
        y1 = max(cy - pad_y, 0)
        y2 = min(cy + pad_y, h_img)

        cropped = img_cv[y1:y2, x1:x2]
        user_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA)).convert("RGBA")
        user_img = user_img.resize(out_size)

        bg_hex = (255, 255, 255, 255) if bg_color == "white" else (173, 216, 230, 255)
        bg_img = Image.new("RGBA", out_size, bg_hex)

        result = Image.alpha_composite(bg_img, user_img).convert("RGB")
        filename = f"{file.filename}_passport.jpg"
        path = f"output/{filename}"
        result.save(path, format="JPEG", quality=95)

        return templates.TemplateResponse("passport.html", {
            "request": request,
            "result_url": f"/output/{filename}"
        })

    except Exception as e:
        return templates.TemplateResponse("passport.html", {
            "request": request,
            "error": f"Error occurred: {e}"
        })

# ========= ROUTE: AI Headshot Generator =========

@app.get("/headshot", response_class=HTMLResponse)
async def get_headshot_form(request: Request):
    return templates.TemplateResponse("headshot.html", {"request": request})

@app.post("/headshot")
async def generate_headshot(
    request: Request,
    file: UploadFile = File(...),
    bg: str = Form("white")
):
    input_bytes = await file.read()

    try:
        no_bg = remove(input_bytes)
        img_array = np.frombuffer(no_bg, np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        face_cascade = cv2.CascadeClassifier("app/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return templates.TemplateResponse("headshot.html", {
                "request": request,
                "error": "No face found in image."
            })

        x, y, w, h = faces[0]
        pad_top = int(h * 1.2)
        pad_bottom = int(h * 1.8)
        pad_side = int(w * 0.8)
        height, width = img_cv.shape[:2]
        cx, cy = x + w // 2, y + h // 2
        x1 = max(cx - pad_side, 0)
        x2 = min(cx + pad_side, width)
        y1 = max(cy - pad_top, 0)
        y2 = min(cy + pad_bottom, height)
        cropped = img_cv[y1:y2, x1:x2]

        user_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA)).convert("RGBA")
        user_img = user_img.resize((600, 800))

        if bg == "white":
            bg_img = Image.new("RGBA", user_img.size, (255, 255, 255, 255))
        elif bg == "blur":
            blur_cv = cv2.GaussianBlur(cropped, (51, 51), 0)
            bg_img = Image.fromarray(cv2.cvtColor(blur_cv, cv2.COLOR_BGR2RGBA)).resize((600, 800))
        elif bg == "office":
            bg_path = "static/office_bg.jpg"
            if os.path.exists(bg_path):
                bg_img = Image.open(bg_path).convert("RGBA").resize((600, 800))
            else:
                bg_img = Image.new("RGBA", user_img.size, (240, 240, 240, 255))
        else:
            bg_img = Image.new("RGBA", user_img.size, (240, 240, 240, 255))

        final = Image.alpha_composite(bg_img, user_img).convert("RGB")
        final = ImageEnhance.Contrast(final).enhance(1.1)
        final = ImageEnhance.Sharpness(final).enhance(1.2)
        final = ImageEnhance.Brightness(final).enhance(1.05)

        filename = f"{file.filename}_headshot.jpg"
        out_path = f"output/{filename}"
        final.save(out_path, format="JPEG", quality=85)

        return templates.TemplateResponse("headshot.html", {
            "request": request,
            "result_url": f"/output/{filename}"
        })

    except Exception as e:
        return templates.TemplateResponse("headshot.html", {
            "request": request,
            "error": f"Something went wrong: {e}"
        })

    return content
# ========= STATIC FILES =========

app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")
