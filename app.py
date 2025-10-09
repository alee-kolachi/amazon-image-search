# app.py
import os
import re
import io
import time
import base64
import logging
from typing import Optional
from urllib.parse import urlparse, unquote

import requests
from flask import Flask, request, jsonify
from google.cloud import vision
from groq import Groq
from PIL import Image, ExifTags

# --- Configurable constants ---
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1024"))  # resize long side to this (px)
VISION_MIN_CONFIDENCE = float(os.getenv("VISION_MIN_CONFIDENCE", "0.40"))  # label confidence threshold
PAGE_FETCH_TIMEOUT = float(os.getenv("PAGE_FETCH_TIMEOUT", "5"))
IMAGE_FETCH_TIMEOUT = float(os.getenv("IMAGE_FETCH_TIMEOUT", "10"))
HTTP_RETRY_COUNT = int(os.getenv("HTTP_RETRY_COUNT", "2"))

# --- Logging ---
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("app")

app = Flask(__name__)

# --- Initialize Google Vision client ---
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_APPLICATION_CREDENTIALS:
    raise ValueError("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable with path to your JSON key.")
vision_client = vision.ImageAnnotatorClient()

# --- Initialize Groq client ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY environment variable.")
client = Groq(api_key=GROQ_API_KEY)

# ----------------- Utility functions -----------------
def get_image_bytes(image_url: Optional[str] = None, image_base64: Optional[str] = None) -> bytes:
    """
    Fetch image bytes either from a URL or a base64 string.
    Raises ValueError with contextual message on failure.
    """
    if image_url:
        headers = {"User-Agent": "product-title-bot/1.0"}
        last_exc = None
        for attempt in range(HTTP_RETRY_COUNT + 1):
            try:
                r = requests.get(image_url, headers=headers, timeout=IMAGE_FETCH_TIMEOUT)
                r.raise_for_status()
                return r.content
            except Exception as e:
                last_exc = e
                log.warning("Attempt %s: failed to fetch image '%s': %s", attempt + 1, image_url, e)
                time.sleep(0.3)
        raise ValueError(f"Failed to fetch image from URL after {HTTP_RETRY_COUNT + 1} attempts: {last_exc}")
    if image_base64:
        try:
            b = re.sub(r"^data:image/\w+;base64,", "", image_base64)
            return base64.b64decode(b)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")
    raise ValueError("Either image_url or image_base64 must be provided.")

def preprocess_image_bytes(img_bytes: bytes, max_side: int = MAX_IMAGE_SIDE) -> bytes:
    """
    Use Pillow to fix orientation and resize the image so that the long side <= max_side.
    Return JPEG bytes suitable for Vision API.
    """
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            # Fix EXIF orientation if present
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = im._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        im = im.rotate(180, expand=True)
                    elif orientation_value == 6:
                        im = im.rotate(270, expand=True)
                    elif orientation_value == 8:
                        im = im.rotate(90, expand=True)
            except Exception:
                # silently ignore EXIF issues
                pass

            # Resize if too large
            w, h = im.size
            long_side = max(w, h)
            if long_side > max_side:
                scale = max_side / float(long_side)
                new_size = (int(w * scale), int(h * scale))
                im = im.resize(new_size, Image.LANCZOS)

            # Convert to RGB and save as JPEG
            if im.mode != "RGB":
                im = im.convert("RGB")
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=88, optimize=True)
            return out.getvalue()
    except Exception as e:
        log.warning("Image preprocessing failed, returning original bytes: %s", e)
        return img_bytes  # fallback

def clean_page_url_tokens(page_url: str):
    try:
        p = urlparse(page_url)
        host = re.sub(r"^www\.", "", p.netloc or "", flags=re.I).replace(".", " ")
        path = p.path or ""
        segments = [unquote(s) for s in path.split("/") if s]
        combined = f"{host} {' '.join(segments)}"
        combined = re.sub(r"[_\-]+", " ", combined)
        combined = re.sub(r"\s+", " ", combined).strip()
        return combined
    except Exception:
        return ""

def extract_page_product_tokens(page_url: str):
    try:
        p = urlparse(page_url)
        segments = [unquote(s) for s in p.path.split("/") if s]
        host = re.sub(r"^www\.", "", p.netloc or "")
        tokens = " ".join([host] + segments)
        tokens = re.sub(r'[\d]+', '', tokens)
        tokens = re.sub(r'\.html|\.php', '', tokens)
        tokens = re.sub(r'[-_]+', ' ', tokens)
        tokens = re.sub(r'\s+', ' ', tokens).strip()
        return tokens
    except:
        return ""

def clean_url_tokens(image_url: str):
    try:
        p = urlparse(image_url)
        host = re.sub(r"^www\.", "", p.netloc or "", flags=re.I).replace(".", " ")
        path = p.path or ""
        segments = [unquote(s) for s in path.split("/") if s]
        file_part = segments[-1] if segments else ""
        base = re.sub(r"\.\w+$", "", file_part)
        combined = f"{host} {' '.join(segments)} {base}"
        combined = re.sub(r"[_\-]+", " ", combined)
        combined = re.sub(r"\b(AC|SX|UX)\d+\b", "", combined)
        return re.sub(r"\s+", " ", combined).strip()
    except Exception:
        return ""

def clean_final_title(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"https?://\S+|www\.\S+|\bamazon\.com\b", "", s, flags=re.I)
    s = re.sub(r'[^A-Za-z0-9 \-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def fallback_title(caption, url_tokens, page_url_tokens, page_text):
    first_label = caption.split(",")[0] if caption else ""
    parts = [first_label, url_tokens, page_url_tokens, page_text]
    combined = " ".join([p for p in parts if p])
    combined = combined.strip()[:120]
    return clean_final_title(combined)

def safe_extract_title_from_groq_stream_or_resp(resp_iterable_or_obj) -> str:
    """
    Accept either a streaming iterator of chunks or a non-stream response object.
    Try several shapes safely and return the final text string.
    """
    raw_text = ""
    # If iterable (streaming)
    try:
        for chunk in resp_iterable_or_obj:
            try:
                delta = chunk.choices[0].delta
                raw_text += delta.content or ""
            except Exception:
                # skip malformed chunk
                continue
    except TypeError:
        # Not iterable: try to parse as single response object
        try:
            if resp_iterable_or_obj and getattr(resp_iterable_or_obj, "choices", None):
                choice = resp_iterable_or_obj.choices[0]
                if hasattr(choice, "message") and getattr(choice.message, "content", None):
                    raw_text = choice.message.content
                elif getattr(choice, "text", None):
                    raw_text = choice.text
        except Exception:
            raw_text = ""
    except Exception:
        raw_text = ""
    return raw_text or ""

# ----------------- CORS -----------------
@app.after_request
def cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

# ----------------- Main endpoint -----------------
@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True, silent=True) or {}
    image_url = data.get("imageUrl")
    page_url = data.get("pageUrl", "")
    image_base64 = data.get("imageBase64")
    page_url_tokens = clean_page_url_tokens(page_url) if page_url else ""
    log.info("Page URL Tokens: %s", page_url_tokens)

    if not image_url and not image_base64:
        return jsonify({"error": "imageUrl or imageBase64 required"}), 400

    try:
        # 1️⃣ Load image
        raw_bytes = get_image_bytes(image_url, image_base64)
        if not raw_bytes:
            raise ValueError("Could not load image bytes")

        # 1.1 Preprocess (resize, fix orientation)
        img_bytes = preprocess_image_bytes(raw_bytes)

        # 2️⃣ Google Vision: try label_detection first (non-fatal)
        labels = []
        labels_with_confidence = []
        caption = "Unknown product"
        vision_method = "label_detection"
        try:
            image = vision.Image(content=img_bytes)
            response = vision_client.label_detection(image=image, max_results=10)
            # handle low-level error
            if getattr(response, "error", None) and getattr(response.error, "message", None):
                log.warning("Vision API returned error: %s", response.error.message)
            else:
                if getattr(response, "label_annotations", None):
                    for lab in response.label_annotations:
                        # candidate: label.description, lab.score (0..1)
                        name = getattr(lab, "description", None)
                        score = float(getattr(lab, "score", 0.0))
                        labels_with_confidence.append({"name": name, "confidence": score})
                        if score >= VISION_MIN_CONFIDENCE:
                            labels.append(name)
                if labels:
                    caption = ", ".join(labels)
                else:
                    # keep best label if exists even if below threshold as fallback
                    if labels_with_confidence:
                        best = sorted(labels_with_confidence, key=lambda x: x["confidence"], reverse=True)[0]
                        caption = best["name"] if best.get("name") else caption
        except Exception as e:
            log.warning("Warning: Google Vision label_detection failed: %s", e)

        # If label detection gave nothing or low confidence, try web_detection as fallback
        if (not labels) and getattr(response, "label_annotations", None):
            # already looked at label annotations but low confidences => attempt web_detection
            try:
                vision_method = "web_detection"
                web_resp = vision_client.web_detection(image=image)
                web = getattr(web_resp, "web_detection", None)
                web_entities = []
                if web:
                    if getattr(web, "best_guess_labels", None):
                        web_entities.extend([b.label for b in web.best_guess_labels if getattr(b, "label", None)])
                    if getattr(web, "web_entities", None):
                        web_entities.extend([we.description for we in web.web_entities if getattr(we, "description", None)])
                web_entities = [w for w in web_entities if w]
                if web_entities:
                    caption = ", ".join(web_entities[:5])
                    labels = web_entities[:5]
            except Exception as e:
                log.info("web_detection fallback failed or returned nothing: %s", e)

        log.info("Vision method: %s; caption: %s", vision_method, caption)

        # 3️⃣ Extract URL tokens
        url_tokens = clean_url_tokens(image_url) if image_url else ""
        log.debug("URL Tokens: %s", url_tokens)

        # 4️⃣ Page fetch (non-fatal) and extract title
        page_text = ""
        page_url_tokens = extract_page_product_tokens(page_url) if page_url else ""
        if page_url:
            headers = {"User-Agent": "product-title-bot/1.0"}
            last_exc = None
            for attempt in range(HTTP_RETRY_COUNT + 1):
                try:
                    r = requests.get(page_url, headers=headers, timeout=PAGE_FETCH_TIMEOUT)
                    r.raise_for_status()
                    m = re.search(r'<title>(.*?)</title>', r.text, re.I | re.S)
                    if m:
                        page_text = m.group(1).strip()
                    break
                except Exception as e:
                    last_exc = e
                    log.warning("Attempt %s: couldn't fetch page_url '%s': %s", attempt + 1, page_url, e)
                    time.sleep(0.2)
            if not page_text and last_exc:
                log.info("Page fetch gave no title (last error: %s)", last_exc)

        # 5️⃣ Groq LLM prompt to build product title
        system_msg = {
            "role": "system",
            "content": (
                "You are an assistant that must produce a single concise product title suitable for Amazon search. "
                "Use the image caption, the URL tokens, and the page URL for context. "
                "If you know the product title, output it. Otherwise, enhance the caption and URL tokens into a concise title. "
                "Output ONLY the final title."
            )
        }
        user_msg = {
            "role": "user",
            "content": (
                f"Image caption: {caption}\n"
                f"Image URL tokens: {url_tokens}\n"
                f"Page URL tokens: {page_url_tokens}\n"
                f"Page title or context: {page_text}\n"
                "Return a single product title optimized for Amazon search, "
                "prioritize words from the URL if they clearly indicate the product. Use product codes/size/color/style or anything that clearly defined the product. You should add features in the title and not details"
            )
        }

        raw_text = ""
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[system_msg, user_msg],
                temperature=0.2,
                max_completion_tokens=60,
                top_p=1,
                stream=True
            )
            raw_text = safe_extract_title_from_groq_stream_or_resp(completion)
        except Exception as e:
            log.warning("Warning: Groq streaming failed: %s", e)
            # Try non-streaming as fallback
            try:
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[system_msg, user_msg],
                    temperature=0.2,
                    max_completion_tokens=60,
                    top_p=1,
                    stream=False
                )
                raw_text = safe_extract_title_from_groq_stream_or_resp(resp)
            except Exception as e2:
                log.warning("Warning: Groq non-streaming fallback also failed: %s", e2)
                raw_text = ""

        log.debug("Raw Groq output: %s", raw_text)

        # Clean and finalize title
        final_title = clean_final_title(raw_text)
        if not final_title:
            final_title = fallback_title(caption, url_tokens, page_url_tokens, page_text)
            log.info("Using fallback title: %s", final_title)

        log.info("Final Clean Title: %s", final_title)

        return jsonify({
            "caption": caption,
            "vision_method": vision_method,
            "labels_with_confidence": labels_with_confidence,
            "url_tokens": url_tokens,
            "page_url": page_url,
            "title": final_title
        }), 200

    except Exception as e:
        log.exception("Error in analyze (unexpected): %s", e)
        return jsonify({"error": str(e)}), 500


# --- Run ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=os.getenv("FLASK_DEBUG", "False") == "True")
