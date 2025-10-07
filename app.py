# app.py
import os
import re
import base64
import requests
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse, unquote
from transformers import pipeline
from groq import Groq

app = Flask(__name__)

# --- Initialize BLIP caption model ---
print("Loading BLIP caption model...")
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
print("Caption model ready.")

# --- Initialize Groq client ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY environment variable.")
client = Groq(api_key=GROQ_API_KEY)

# --- Utility functions ---
def get_image_bytes(image_url=None, image_base64=None):
    if image_url:
        r = requests.get(image_url, timeout=10)
        r.raise_for_status()
        return r.content
    if image_base64:
        b = re.sub(r"^data:image/\\w+;base64,", "", image_base64)
        return base64.b64decode(b)
    return None


def clean_page_url_tokens(page_url: str):
    try:
        p = urlparse(page_url)
        host = re.sub(r"^www\\.", "", p.netloc or "", flags=re.I).replace(".", " ")
        path = p.path or ""
        segments = [unquote(s) for s in path.split("/") if s]
        combined = f"{host} {' '.join(segments)}"
        combined = re.sub(r"[_\\-]+", " ", combined)
        combined = re.sub(r"\\s+", " ", combined).strip()
        return combined
    except Exception:
        return ""

def extract_page_product_tokens(page_url: str):
    try:
        p = urlparse(page_url)
        # Take host + path
        segments = [unquote(s) for s in p.path.split("/") if s]
        host = re.sub(r"^www\\.", "", p.netloc or "")
        tokens = " ".join([host] + segments)
        # Remove common URL noise like numeric IDs, file extensions
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
        host = re.sub(r"^www\\.", "", p.netloc or "", flags=re.I).replace(".", " ")
        path = p.path or ""
        segments = [unquote(s) for s in path.split("/") if s]
        file_part = segments[-1] if segments else ""
        base = re.sub(r"\\.\w+$", "", file_part)
        combined = f"{host} {' '.join(segments)} {base}"
        combined = re.sub(r"[_\\-]+", " ", combined)
        combined = re.sub(r"\\b(AC|SX|UX)\\d+\\b", "", combined)
        return re.sub(r"\\s+", " ", combined).strip()
    except Exception:
        return ""

def clean_final_title(s):
    if not s:
        return ""
    s = re.sub(r"https?://\S+|www\.\S+|\bamazon\.com\b", "", s, flags=re.I)
    s = re.sub(r'[^A-Za-z0-9 \-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

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
        print("Received image URL:", image_url)
        print("Received page URL:", page_url)

        # 1️⃣ Load image
        img_bytes = get_image_bytes(image_url, image_base64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # 2️⃣ BLIP caption
        caption = captioner(img)[0]["generated_text"].strip()
        print("BLIP Caption:", caption)

        # 3️⃣ Extract URL tokens
        url_tokens = clean_url_tokens(image_url) if image_url else ""
        print("URL Tokens:", url_tokens)

        # 4️⃣ Groq LLM prompt
        system_msg = {
            "role": "system",
            "content": (
                "You are an assistant that must produce a single concise product title suitable for Amazon search. "
                "Use the image caption, the URL tokens, and the page URL for context. "
                "If you know the product title, output it. Otherwise, enhance the caption and URL tokens into a concise title. "
                "Output ONLY the final title."
            )
        }
        page_url_tokens = extract_page_product_tokens(page_url)
        page_text = ""
        if page_url:
            r = requests.get(page_url, timeout=5)
            r.raise_for_status()
            # simple extraction of <title>
            m = re.search(r'<title>(.*?)</title>', r.text, re.I)
            if m:
                page_text = m.group(1)

        user_msg = {
            "role": "user",
            "content": (
                f"Image caption: {caption}\n"
                f"Image URL tokens: {url_tokens}\n"
                f"Page URL tokens: {page_url_tokens}\n"
                f"Page title or context: {page_text}\n"
                "Return a single concise product title optimized for Amazon search, "
                "prioritize words from the URL if they clearly indicate the product. Use product codes/size/color/style or anything that clearly defined the product. The goal is to describe the product in a best way."
            )
        }



        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[system_msg, user_msg],
            temperature=0.2,
            max_completion_tokens=60,
            top_p=1,
            stream=True
        )

        # 5️⃣ Collect streaming text
        raw_text = ""
        for chunk in completion:
            raw_text += chunk.choices[0].delta.content or ""
        print("Raw Groq output:", raw_text)

        # 6️⃣ Post-process
        final_title = clean_final_title(raw_text)
        print("Final Clean Title:", final_title)

        return jsonify({
            "caption": caption,
            "url_tokens": url_tokens,
            "page_url": page_url,
            "title": final_title
        })

    except Exception as e:
        print("Error in analyze:", e)
        return jsonify({"error": str(e)}), 500

# --- Run ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
