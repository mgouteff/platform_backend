# embedding_to_supabase.py
import os
import re
import uuid
import csv
import pdfplumber
from datetime import datetime
from dotenv import load_dotenv
import concurrent.futures
from openai import OpenAI

# Local services
from services.embeddings import get_embedding
from services.supabase_client import supabase

# ----------------------------
# Setup
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# GPT Prompt Templates
# ----------------------------
CHUNK_PROMPT = """You are an expert content editor. Split the following text into logical chunks of 625–850 words each,
ideally around 725 words. Do not split mid-sentence. Each chunk must strictly follow this format:

Title: <short descriptive title>
Tags: <comma-separated keywords>
Content: <chunk body>

Text to split:
{block}
"""

CHUNK_REPAIR_PROMPT_TEMPLATE = """You are an expert editor. The following chunk is not within the required 500–850 word range.
Revise it so the content is complete, meaningful, and lands between 625–850 words. Preserve nuance and examples.
Maintain this exact format:

Title: <short descriptive title>
Tags: <comma-separated keywords>
Content: <chunk body>

Feedback: {feedback}

Chunk:
{block}
"""

# ----------------------------
# GPT Helpers
# ----------------------------
def call_gpt_with_timeout(prompt, timeout=90):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
        )
        return future.result(timeout=timeout).choices[0].message.content

def chunk_text(block):
    prompt = CHUNK_PROMPT.format(block=block)
    return call_gpt_with_timeout(prompt)

def repair_chunk(block, feedback=""):
    prompt = CHUNK_REPAIR_PROMPT_TEMPLATE.format(block=block, feedback=feedback)
    return call_gpt_with_timeout(prompt)

def parse_chunks(text):
    chunks, current = [], {"title": "", "tags": "", "content": ""}
    section = None
    for line in text.splitlines():
        if line.startswith("Title:"):
            if current["content"]:
                chunks.append(current)
                current = {"title": "", "tags": "", "content": ""}
            current["title"] = line.replace("Title:", "").strip()
            section = "title"
        elif line.startswith("Tags:"):
            current["tags"] = line.replace("Tags:", "").strip()
            section = "tags"
        elif line.startswith("Content:"):
            section = "content"
        else:
            if section == "content":
                current["content"] += line.strip() + " "
    if current["content"]:
        chunks.append(current)
    return chunks

def is_valid_chunk(chunk):
    words = chunk["content"].split()
    return 500 <= len(words) <= 850

# ----------------------------
# Detection Logic
# ----------------------------
def detect_method(text, filename=None):
    if filename:
        if re.search(r"(law|contract|code|act|agreement|policy)", filename.lower()):
            return "structure"
        if re.search(r"(transcript|interview|memo|minutes)", filename.lower()):
            return "meaning"

    if re.search(r"(§\d+|Article \d+|Chapter [IVXLC]+)", text):
        return "structure"
    if re.search(r"^\w+:\s", text, flags=re.MULTILINE):
        return "meaning"
    if re.search(r"(\b\d{1,2}:\d{2}\b|\b\d{1,2}\.\d{2}\b|\b\d{1,2}:\d{2}:\d{2}\b)", text):
        return "meaning"

    return "fixed"

# ----------------------------
# Tagging Helpers
# ----------------------------
def extract_regex_tags(text):
    """Capture universal structural markers (works across languages)."""
    tags = set()
    patterns = [
        r"§\s*\d+[a-zA-Z]*",   # §45, §12a
        r"\b\d+(\.\d+)+\b"     # 1.2, 2.3.4 subsections
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        tags.update([m.strip() for m in matches])
    return list(tags)

def gpt_generate_tags(text, max_semantic=5):
    """
    Use GPT to generate two sets of tags:
    1. Structural references (unlimited, e.g., Článek 12, Hlava IV, Article 5, Paragraph 7)
    2. Semantic meaning tags (limited, e.g., Overtime Pay, Termination Rights)
    """
    prompt = f"""Extract tags from the following text.
Split them into two groups:

1. Structural references (all of them, unlimited). These are numbered legal markers like Článek 12, Hlava IV, Article 5, Paragraph 7, §45, Section 2.
Always return the full reference (word + number/roman numeral).
2. Semantic meaning tags (up to {max_semantic}). These are short topic descriptors like Overtime Pay, Lunch Breaks, Termination Rights.

Output them in this JSON format:
{{
  "structural": ["..."],
  "semantic": ["..."]
}}

Text:
{text[:2000]}
"""
    try:
        output = call_gpt_with_timeout(prompt, timeout=30)

        import json
        parsed = {}
        try:
            parsed = json.loads(output)
        except Exception:
            structural, semantic = [], []
            if "structural" in output.lower():
                structural = re.findall(r"Článek \d+|Hlava [IVXLC]+|Article \d+|Section \d+|Paragraph \d+|§\s*\d+", output)
            if "semantic" in output.lower():
                semantic = [t.strip() for t in output.split(",") if t.strip()]
            parsed = {"structural": structural, "semantic": semantic}

        structural_tags = parsed.get("structural", [])
        semantic_tags = parsed.get("semantic", [])[:max_semantic]

        return structural_tags, semantic_tags

    except Exception as e:
        print(f"⚠️ GPT tagging failed: {e}")
        return [], []

# ----------------------------
# Chunking Methods
# ----------------------------
def chunk_structure(text):
    """Structure-based chunking with hybrid tagging (regex + GPT)."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk, current_len = [], 0
    chunk_size = 800

    for para in paragraphs:
        para_len = len(para.split())
        if current_len + para_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    chunk_dicts = []
    for c in chunks:
        regex_tags = extract_regex_tags(c)
        gpt_structural, gpt_semantic = gpt_generate_tags(c, max_semantic=5)
        combined_tags = regex_tags + gpt_structural + gpt_semantic

        chunk_dicts.append({
            "title": "General",
            "tags": ", ".join(combined_tags),
            "content": c
        })

    return chunk_dicts

def chunk_meaning(text):
    """Semantic chunking with GPT (preserves original content)."""
    blocks = [text[i:i+5000] for i in range(0, len(text), 5000)]
    all_chunks = []

    for block in blocks:
        chunked = parse_chunks(chunk_text(block))

        repaired_chunks = []
        for ch in chunked:
            if not is_valid_chunk(ch):
                repaired = repair_chunk(
                    ch["content"],
                    feedback=f"Chunk had {len(ch['content'].split())} words. Adjust to 625–850."
                )
                new_chunks = parse_chunks(repaired)
                if new_chunks and is_valid_chunk(new_chunks[0]):
                    repaired_chunks.append(new_chunks[0])
                else:
                    repaired_chunks.append(ch)
            else:
                repaired_chunks.append(ch)

        all_chunks.extend(repaired_chunks)

    return all_chunks

def chunk_fixed(text, chunk_size=650, overlap=150, max_semantic=5):
    """Fixed-size chunking with overlap + GPT semantic tagging."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        content = " ".join(chunk)

        _, gpt_semantic = gpt_generate_tags(content, max_semantic=max_semantic)

        chunks.append({
            "title": f"Chunk {i+1}",
            "tags": ", ".join(gpt_semantic),
            "content": content
        })

    return chunks

# ----------------------------
# Main ingestion function
# ----------------------------
def ingest_file(file_path: str, force_method: str = None):
    filename = os.path.basename(file_path)

    text = ""
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    method = force_method or detect_method(text, filename)
    print(f"📄 File: {filename} → Method: {method}")

    if method == "structure":
        chunks = chunk_structure(text)
    elif method == "meaning":
        chunks = chunk_meaning(text)
    else:
        chunks = chunk_fixed(text)

    print(f"✂️ Created {len(chunks)} chunks.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"{os.path.splitext(filename)[0]}_{method}_{timestamp}.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "content", "tags", "method", "source_file"])
        for idx, ch in enumerate(chunks, 1):
            chunk_id = str(uuid.uuid4())
            title = ch.get("title") or f"{filename} - chunk {idx}"
            tags = ch.get("tags", "")
            content = ch["content"]

            embedding = get_embedding(content)
            supabase.table("documents").insert({
                "id": chunk_id,
                "title": title,
                "content": content,
                "tags": tags,
                "method": method,
                "source_file": filename,
                "embedding": embedding
            }).execute()

            writer.writerow([chunk_id, title, content, tags, method, filename])

    print(f"✅ Inserted {len(chunks)} chunks into Supabase")
    print(f"📂 CSV saved: {csv_file}")

# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("❌ Usage: python embedding_to_supabase.py <file> [force_method]")
        sys.exit(1)
    file_path = sys.argv[1]
    force_method = sys.argv[2] if len(sys.argv) > 2 else None
    ingest_file(file_path, force_method)


