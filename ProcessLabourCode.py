import pdfplumber
import re
import csv

# ---------------------------
# CONFIG
# ---------------------------
INPUT_PDF = r"C:\Users\mgout\Documents\AI Accounts App\Raw Data\Zakonik_prace_consolidated_June2025.pdf"
OUTPUT_TXT = r"C:\Users\mgout\Documents\AI Accounts App\Processed Data\labour_code_cleaned.txt"
OUTPUT_CSV = r"C:\Users\mgout\Documents\AI Accounts App\Processed Data\labour_code_chunks.csv"

CHUNK_SIZE = 800   # target chunk size in words (500–1000 window)

# ---------------------------
# STEP 1: Extract text from PDF
# ---------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---------------------------
# STEP 2: Clean the text
# ---------------------------
def clean_text(text):
    # remove page numbers (standalone digits)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # remove timestamps like "9/25/25, 8:33 PM"
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(AM|PM)', '', text)

    # remove URLs
    text = re.sub(r'http\S+', '', text)

    # remove artefacts like "zakonyprolidi_cs_2006_262_v20250601"
    text = re.sub(r'zakonyprolidi[_a-z0-9]+', '', text, flags=re.IGNORECASE)

    # remove page markers like "1/204", "23/204"
    text = re.sub(r'\b\d+/\d+\b', '', text)

    # collapse multiple blank lines and extra spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()

# ---------------------------
# STEP 3: Split into paragraphs
# ---------------------------
def split_into_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs

# ---------------------------
# STEP 4: Group paragraphs into chunks
# ---------------------------
def group_paragraphs(paragraphs, chunk_size=CHUNK_SIZE):
    chunks = []
    current_chunk = []
    current_len = 0

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

    return chunks

# ---------------------------
# STEP 5: Assign titles
# ---------------------------
def assign_titles(chunks):
    title_pattern = re.compile(
        r'(§\s*\d+\s*[A-Za-zÁ-ž ]{0,50}|Hlava\s+[IVXL]+|ČÁST\s+[IVXL]+|Oddíl\s+[IVXL]+)',
        re.IGNORECASE
    )
    results = []
    current_title = "General"

    for chunk in chunks:
        match = title_pattern.search(chunk)
        if match:
            current_title = match.group().strip()
        results.append((current_title, chunk))

    return results

# ---------------------------
# STEP 6: Auto-tagging
# ---------------------------
def auto_tag(text):
    tags = []
    rules = {
        "pracovní poměr": "employment contract",
        "pracovní doba": "working time",
        "dovolená": "vacation",
        "mzda": "wages",
        "platy": "salaries",
        "pracoviště": "workplace",
        "pracovní úraz": "work injury",
        "bezpečnost": "safety",
        "odstupné": "severance",
        "dohoda": "agreement",
        "práce na dálku": "remote work",
        "sdílené pracovní místo": "job sharing"
    }
    for keyword, tag in rules.items():
        if keyword in text.lower():
            tags.append(tag)
    return sorted(set(tags))

# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    print("📥 Extracting text from PDF...")
    raw_text = extract_text_from_pdf(INPUT_PDF)

    print("🧹 Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("📑 Splitting into paragraphs...")
    paragraphs = split_into_paragraphs(cleaned_text)

    print("✂️ Grouping paragraphs into chunks...")
    chunks = group_paragraphs(paragraphs)

    print("📝 Assigning titles...")
    titled_chunks = assign_titles(chunks)

    print("🏷️ Auto-tagging...")
    rows = []
    for (title, chunk_text) in titled_chunks:
        tags = auto_tag(chunk_text)
        # Convert tags list into Postgres array format
        tags_array = "{" + ",".join(tags) + "}" if tags else "{}"
        rows.append([title, tags_array, chunk_text, "labour_code"])

    print("💾 Saving outputs...")
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["title", "tags", "chunk_text", "source_ref"])
        writer.writerows(rows)

    print(f"✅ Done! Extracted {len(rows)} chunks.")
    print(f"   Cleaned text saved to: {OUTPUT_TXT}")
    print(f"   Chunks saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
