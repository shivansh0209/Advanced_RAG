"""
PDF Health Check Script for RAG Dataset
========================================
Checks each PDF for:
  - File existence
  - Page count & file size
  - Font presence (scanned vs text-based)
  - Embedded font quality (encoding risks)
  - Text extractability & yield
  - Garbled text detection
  - Correct act content (title keyword match)

Usage:
    python check_pdfs.py

Requirements:
    pip install pypdf pdfplumber
    sudo apt install poppler-utils   # for pdfinfo, pdffonts, pdftotext
"""

import os
import re
import subprocess
import unicodedata

pdfplumber = None

# ── Configure your PDFs here ─────────────────────────────────────────────────
ACT_NAMES = {
    "data/crpc_1973.pdf":        "code of criminal procedure, 1973",  # rename CRPC1973.pdf
    "data/rti_act_2005.pdf":         "right to information act, 2005",    # rename aa2005.pdf        ← still needs new PDF (6.9% garble)
    "data/A1955-25.pdf":         "hindu marriage act, 1955",          # rename A1955-25Eng.pdf   ← still needs new PDF (6.9% garble)
    "data/it_act_2000.pdf":      "information technology act, 2000",  # already clean name ✅
    "data/cpa_2019.pdf":         "consumer protection act, 2019",     # rename cpa.pdf
    "data/contract.pdf":         "indian contract act, 1872",        # new PDF needed
    "data/ipc_act.pdf":         "indian penal code, 1860",           # new PDF needed
}

# Keywords expected near the top of each act (used for content verification)
ACT_KEYWORDS = {
    "data/crpc_1973.pdf":       ["criminal procedure", "magistrate", "cognizable"],
    "data/rti_act_2005.pdf":         ["right to information", "public information officer", "disclosure"],
    "data/A1955-25.pdf":    ["hindu marriage", "solemnization", "matrimonial"],
    "data/it_act_2000.pdf":    ["information technology", "electronic", "digital signature"],
    "data/contract.pdf":      ["contract", "agreement", "liability"],
    "data/ipc_act.pdf":      ["penal code", "punishment", "offence"],
    "data/cpa_2019.pdf":      ["consumer protection", "complaint", "redressal"],
}

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✔ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠ {msg}{RESET}")
def fail(msg):  print(f"  {RED}✘ {msg}{RESET}")
def info(msg):  print(f"    {msg}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd):
    """Run a shell command and return (stdout, stderr, returncode)."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def garble_ratio(text):
    """
    Fraction of characters that are non-printable / replacement chars /
    control characters — a proxy for broken encoding.
    """
    if not text:
        return 1.0
    bad = sum(
        1 for c in text
        if unicodedata.category(c) in ("Cc", "Cs") or c == "\ufffd"
    )
    return bad / len(text)


def words_per_page(text, pages):
    if not pages or not text:
        return 0
    return len(text.split()) / pages


def check_fonts(path):
    """
    Returns (has_fonts, has_risky_fonts, font_lines).
    Risky = not embedded AND unicode=no  →  garbled extraction likely.
    """
    out, _, _ = run(f'pdffonts "{path}"')
    lines = out.splitlines()
    data_lines = [l for l in lines if l and not l.startswith("name") and not l.startswith("---")]
    if not data_lines:
        return False, False, []

    risky = []
    for l in data_lines:
        parts = l.split()
        if len(parts) >= 6:
            emb = parts[-4]   # emb column
            uni = parts[-2]   # uni column
            if emb == "no" and uni == "no":
                risky.append(l)

    return True, bool(risky), data_lines


def extract_text_sample(path, pages=3):
    """Extract text from first N pages using pdfplumber (best for legal docs)."""
    if pdfplumber is None:
        out, _, _ = run(f'pdftotext -f 1 -l {pages} "{path}" -')
        return out
    try:
        with pdfplumber.open(path) as pdf:
            text = ""
            for page in pdf.pages[:pages]:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text
    except Exception as e:
        return ""


def get_page_count(path):
    out, _, _ = run(f'pdfinfo "{path}"')
    for line in out.splitlines():
        if line.lower().startswith("pages:"):
            return int(line.split(":")[1].strip())
    return 0


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def check_content(text, keywords):
    """Check that expected keywords appear in the extracted text."""
    text_lower = text.lower()
    found     = [kw for kw in keywords if kw in text_lower]
    missing   = [kw for kw in keywords if kw not in text_lower]
    return found, missing


# ── Main check ────────────────────────────────────────────────────────────────

def check_pdf(path, act_name):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}📄 {act_name.upper()}{RESET}")
    print(f"   File: {path}")

    issues = 0

    # 1. File existence
    if not os.path.exists(path):
        fail(f"File not found: {path}")
        return {"file": path, "act": act_name, "status": "MISSING", "issues": 99}
    ok("File exists")

    # 2. File size
    size_mb = get_file_size_mb(path)
    if size_mb < 0.05:
        fail(f"File suspiciously small: {size_mb:.2f} MB")
        issues += 1
    elif size_mb > 50:
        warn(f"File very large: {size_mb:.1f} MB — chunking may be slow")
    else:
        ok(f"File size: {size_mb:.2f} MB")

    # 3. Page count
    pages = get_page_count(path)
    if pages == 0:
        fail("Could not determine page count — file may be corrupt")
        issues += 1
    elif pages < 5:
        warn(f"Very few pages ({pages}) — may be an amendment or partial act")
        issues += 1
    else:
        ok(f"Page count: {pages} pages")

    # 4. Font check (scanned vs text-based)
    has_fonts, has_risky, font_lines = check_fonts(path)
    if not has_fonts:
        fail("No fonts detected — PDF appears to be SCANNED (image-only)")
        fail("pdftotext will return empty — OCR required before indexing")
        issues += 3
    else:
        ok(f"Fonts detected ({len(font_lines)} font entries)")
        if has_risky:
            warn("Some fonts are NOT embedded and have no Unicode mapping")
            warn("→ Text extraction may produce garbled/wrong characters")
            issues += 1

    # 5. Text extraction
    if has_fonts:
        sample = extract_text_sample(path, pages=3)
        word_count = len(sample.split())

        if word_count < 50:
            fail(f"Very little text extracted from first 3 pages ({word_count} words)")
            fail("→ Likely encoding issue or effectively scanned PDF")
            issues += 2
        elif word_count < 200:
            warn(f"Low text yield from first 3 pages ({word_count} words) — check extraction")
            issues += 1
        else:
            ok(f"Text extraction yield: ~{word_count} words from first 3 pages")

        # 6. Garble detection
        ratio = garble_ratio(sample)
        if ratio > 0.05:
            fail(f"High garble ratio: {ratio:.1%} of characters are invalid/replacement")
            fail("→ Encoding is broken — extraction will be unreliable")
            issues += 2
        elif ratio > 0.01:
            warn(f"Mild garble ratio: {ratio:.1%} — minor encoding issues possible")
            issues += 1
        else:
            ok(f"Garble ratio: {ratio:.1%} — clean text")

        # 7. Content verification (is this the right act?)
        keywords = ACT_KEYWORDS.get(path, [])
        if keywords:
            found, missing = check_content(sample, keywords)
            if not found:
                fail(f"WRONG CONTENT — none of the expected keywords found!")
                fail(f"Expected keywords: {keywords}")
                fail(f"→ This PDF may be the wrong act entirely")
                issues += 3
            elif missing:
                warn(f"Some keywords not in first 3 pages: {missing}")
                warn("→ They may appear later — not necessarily an error")
            else:
                ok(f"Content verified — all keywords found: {found}")

        # 8. Print text preview
        preview = " ".join(sample.split()[:40])
        info(f"Preview: \"{preview}...\"")

    # Summary for this file
    print()
    if issues == 0:
        print(f"  {GREEN}{BOLD}✔ HEALTHY — ready for RAG indexing{RESET}")
    elif issues <= 2:
        print(f"  {YELLOW}{BOLD}⚠ WARNINGS ({issues} issue(s)) — review before indexing{RESET}")
    else:
        print(f"  {RED}{BOLD}✘ UNHEALTHY ({issues} issue(s)) — do NOT index as-is{RESET}")

    return {"file": path, "act": act_name, "issues": issues,
            "pages": pages, "size_mb": round(size_mb, 2)}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  RAG PDF HEALTH CHECK{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")

    if pdfplumber is None:
        warn("pdfplumber not installed — falling back to pdftotext for extraction")
        warn("Install with: pip install pdfplumber")

    results = []
    for path, act_name in ACT_NAMES.items():
        result = check_pdf(path, act_name)
        results.append(result)

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    print(f"  {'Act':<40} {'Pages':>6} {'MB':>6} {'Status'}")
    print(f"  {'─'*40} {'─'*6} {'─'*6} {'─'*12}")

    healthy = warned = unhealthy = missing = 0
    for r in results:
        if r.get("status") == "MISSING":
            status = f"{RED}MISSING{RESET}"
            missing += 1
        elif r["issues"] == 0:
            status = f"{GREEN}HEALTHY{RESET}"
            healthy += 1
        elif r["issues"] <= 2:
            status = f"{YELLOW}WARNINGS{RESET}"
            warned += 1
        else:
            status = f"{RED}UNHEALTHY{RESET}"
            unhealthy += 1

        pages   = r.get("pages", "—")
        size_mb = r.get("size_mb", "—")
        name    = r["act"][:40]
        print(f"  {name:<40} {str(pages):>6} {str(size_mb):>6} {status}")

    print(f"\n  Healthy: {healthy}  |  Warnings: {warned}  |  Unhealthy: {unhealthy}  |  Missing: {missing}")
    print(f"{BOLD}{'═'*60}{RESET}\n")


if __name__ == "__main__":
    main()