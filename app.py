# app.py (simplified and fixed for extension)
import os
import re
import io
import base64
import logging
from urllib.parse import urlparse, unquote
import requests
from flask import Flask, request, jsonify
from google.cloud import vision
from groq import Groq
from PIL import Image, ExifTags

# --- Config ---
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1024"))
VISION_MIN_CONFIDENCE = float(os.getenv("VISION_MIN_CONFIDENCE", "0.4"))
PAGE_FETCH_TIMEOUT = 5.0  # try once only
IMAGE_FETCH_TIMEOUT = 10.0

# --- Logging ---
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("app")

# --- Flask ---
app = Flask(__name__)

# --- Google Vision ---
vision_client = vision.ImageAnnotatorClient()

# --- Groq ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- Utilities ----------------
def get_image_bytes(image_url=None, image_base64=None):
    if image_url:
        try:
            r = requests.get(image_url, timeout=IMAGE_FETCH_TIMEOUT)
            r.raise_for_status()
            return r.content
        except Exception as e:
            raise ValueError(f"Failed to fetch image URL: {e}")
    if image_base64:
        b64 = re.sub(r"^data:image/\w+;base64,", "", image_base64)
        return base64.b64decode(b64)
    raise ValueError("No image provided")

def preprocess_image(img_bytes):
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            # Fix orientation
            try:
                for k, v in ExifTags.TAGS.items():
                    if v == 'Orientation': orientation_key = k
                exif = im._getexif()
                if exif:
                    o = exif.get(orientation_key)
                    if o == 3: im = im.rotate(180, expand=True)
                    elif o == 6: im = im.rotate(270, expand=True)
                    elif o == 8: im = im.rotate(90, expand=True)
            except: pass

            # Resize if large
            w, h = im.size
            if max(w, h) > MAX_IMAGE_SIDE:
                scale = MAX_IMAGE_SIDE / float(max(w, h))
                im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            if im.mode != "RGB":
                im = im.convert("RGB")
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=88, optimize=True)
            return out.getvalue()
    except:
        return img_bytes

def clean_final_title(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"https?://\S+|www\.\S+|\bamazon\.com\b", "", s, flags=re.I)
    s = re.sub(r'[^A-Za-z0-9 \-]', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def safe_extract_title(resp) -> str:
    # Groq streaming or normal response
    raw_text = ""
    try:
        for chunk in resp:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                raw_text += delta.content
    except TypeError:
        try:
            if resp and getattr(resp, "choices", None):
                choice = resp.choices[0]
                if hasattr(choice, "message") and getattr(choice.message, "content", None):
                    raw_text = choice.message.content
                elif getattr(choice, "text", None):
                    raw_text = choice.text
        except: raw_text = ""
    except: raw_text = ""
    return raw_text or ""

def extract_page_title(page_url: str) -> str:
    if not page_url:
        return ""
    try:
        r = requests.get(page_url, headers={"User-Agent": "product-title-bot"}, timeout=PAGE_FETCH_TIMEOUT)
        r.raise_for_status()
        m = re.search(r'<title>(.*?)</title>', r.text, re.I|re.S)
        return m.group(1).strip() if m else ""
    except Exception as e:
        log.warning("Page fetch failed (ignored): %s", e)
        return ""

# ----------------- Endpoint -----------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    image_url = data.get("imageUrl")
    page_url = data.get("pageUrl", "")
    image_base64 = data.get("imageBase64")

    if not image_url and not image_base64:
        return jsonify({"error": "imageUrl or imageBase64 required"}), 400

    try:
        # --- Image processing ---
        img_bytes = preprocess_image(get_image_bytes(image_url, image_base64))

        # --- Google Vision labels ---
        labels, caption = [], "Unknown product"
        try:
            image = vision.Image(content=img_bytes)
            response = vision_client.label_detection(image=image, max_results=10)
            for lab in getattr(response, "label_annotations", []):
                score = float(getattr(lab, "score", 0.0))
                name = getattr(lab, "description", None)
                if score >= VISION_MIN_CONFIDENCE:
                    labels.append(name)
            caption = ", ".join(labels) if labels else caption
        except Exception as e:
            log.warning("Vision failed: %s", e)

        # --- Page title ---
        page_text = extract_page_title(page_url)

        # --- Groq LLM ---
        system_msg = {
            "role": "system",
            "content": "Generate a single concise Amazon product title from image caption, URL tokens, and page title."
        }
        user_msg = {
            "role": "user",
            "content": f"Image caption: {caption}\nPage title: {page_text}\nReturn a single product title optimized for Amazon search."
        }

        final_title = "Product"
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[system_msg, user_msg],
                temperature=0.2,
                max_completion_tokens=60,
                top_p=1,
                stream=False
            )
            raw_text = safe_extract_title(completion)
            final_title = clean_final_title(raw_text) or (labels[0] if labels else "Product")
        except Exception as e:
            log.warning("Groq failed, fallback to Vision labels: %s", e)
            final_title = labels[0] if labels else "Product"

        log.info("Final title: %s", final_title)

        return jsonify({
            "title": final_title,       # <-- guarantees this key exists
            "caption": caption,
            "page_url": page_url
        })

    except Exception as e:
        log.exception("Unexpected error: %s", e)
        return jsonify({"error": str(e)}), 500

# --- Run ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
