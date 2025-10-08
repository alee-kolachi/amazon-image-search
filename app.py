# app.py
import os
import re
import base64
import requests
from flask import Flask, request, jsonify
from urllib.parse import urlparse, unquote
from google.cloud import vision
from groq import Groq

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

# --- Utility functions ---
def get_image_bytes(image_url=None, image_base64=None):
    if image_url:
        try:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            return r.content
        except Exception as e:
            # bubble a clear exception up so caller can decide, but include context
            raise ValueError(f"Failed to fetch image from URL: {e}")
    if image_base64:
        b = re.sub(r"^data:image/\w+;base64,", "", image_base64)
        return base64.b64decode(b)
    return None

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

def clean_final_title(s):
    if not s:
        return ""
    s = re.sub(r"https?://\S+|www\.\S+|\bamazon\.com\b", "", s, flags=re.I)
    s = re.sub(r'[^A-Za-z0-9 \-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Small deterministic fallback builder if LLM fails
def fallback_title(caption, url_tokens, page_url_tokens, page_text):
    # prefer first Vision label, then url tokens, then page text
    first_label = caption.split(",")[0] if caption else ""
    parts = [first_label, url_tokens, page_url_tokens, page_text]
    combined = " ".join([p for p in parts if p])
    # limit length
    combined = combined.strip()[:120]
    return clean_final_title(combined)

# --- CORS ---
@app.after_request
def cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

# --- Main endpoint ---
@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True, silent=True) or {}
    image_url = data.get("imageUrl")
    page_url = data.get("pageUrl", "")
    image_base64 = data.get("imageBase64")
    page_url_tokens = clean_page_url_tokens(page_url) if page_url else ""
    print("Page URL Tokens:", page_url_tokens)

    if not image_url and not image_base64:
        return jsonify({"error": "imageUrl or imageBase64 required"}), 400

    try:
        # 1️⃣ Load image
        img_bytes = get_image_bytes(image_url, image_base64)

        # 2️⃣ Google Vision API labels (non-fatal)
        labels = []
        caption = "Unknown product"
        try:
            image = vision.Image(content=img_bytes)
            response = vision_client.label_detection(image=image)
            # Some clients put errors on response.error
            if getattr(response, "error", None) and getattr(response.error, "message", None):
                print("Vision API returned error:", response.error.message)
            else:
                labels = [label.description for label in response.label_annotations] or []
                if labels:
                    caption = ", ".join(labels)
        except Exception as e:
            print("Warning: Google Vision failed:", e)
            # proceed with empty/unknown caption

        print("Google Vision Labels:", caption)

        # 3️⃣ Extract URL tokens
        url_tokens = clean_url_tokens(image_url) if image_url else ""
        print("URL Tokens:", url_tokens)

        # 4️⃣ Page fetch (non-fatal) and extract title
        page_text = ""
        page_url_tokens = extract_page_product_tokens(page_url) if page_url else ""
        if page_url:
            try:
                r = requests.get(page_url, timeout=5)
                r.raise_for_status()
                m = re.search(r'<title>(.*?)</title>', r.text, re.I)
                if m:
                    page_text = m.group(1)
            except Exception as e:
                # don't fail entire request because page fetch was forbidden / blocked / timed out
                print(f"Warning: couldn't fetch page_url '{page_url}': {e}")

        # 5️⃣ Groq LLM prompt
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
                "prioritize words from the URL if they clearly indicate the product. Use product codes/size/color/style or anything that clearly defined the product. You should add details/features in the title."
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

            for chunk in completion:
                # stream chunks; guard in case chunk lacks expected fields
                try:
                    delta = chunk.choices[0].delta
                    raw_text += delta.content or ""
                except Exception:
                    # skip malformed chunk
                    continue

        except Exception as e:
            print("Warning: Groq streaming failed:", e)
            # Try a non-streaming attempt as a best-effort fallback
            try:
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[system_msg, user_msg],
                    temperature=0.2,
                    max_completion_tokens=60,
                    top_p=1,
                    stream=False
                )
                # depending on response shape, try to extract text
                if resp and getattr(resp, "choices", None):
                    # some SDKs use message/content
                    choice = resp.choices[0]
                    if hasattr(choice, "message") and getattr(choice.message, "content", None):
                        raw_text = choice.message.content
                    elif getattr(choice, "text", None):
                        raw_text = choice.text
            except Exception as e2:
                print("Warning: Groq non-streaming fallback also failed:", e2)
                raw_text = ""

        print("Raw Groq output:", raw_text)

        final_title = clean_final_title(raw_text)
        if not final_title:
            # deterministic fallback so you always return something
            final_title = fallback_title(caption, url_tokens, page_url_tokens, page_text)
            print("Using fallback title:", final_title)

        print("Final Clean Title:", final_title)

        return jsonify({
            "caption": caption,
            "url_tokens": url_tokens,
            "page_url": page_url,
            "title": final_title
        }), 200

    except Exception as e:
        # last-resort: return a useful error without crashing the server process
        print("Error in analyze (unexpected):", e)
        return jsonify({"error": str(e)}), 500

# --- Run ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
