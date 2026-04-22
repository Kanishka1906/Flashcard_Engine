"""
pdf_diagnostic.py  — run this on any PDF to see its font/size structure.
Usage: python pdf_diagnostic.py path/to/file.pdf
"""
import sys, fitz, json
from collections import Counter

def analyse_pdf(path):
    doc = fitz.open(path)
    print(f"\n{'='*60}")
    print(f"PDF: {path}")
    print(f"Pages: {len(doc)}")
    print(f"{'='*60}\n")

    size_counter = Counter()
    flag_counter = Counter()
    samples = {}   # size → list of text samples

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"] != 0:  # 0 = text block
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text or len(text) < 2:
                        continue
                    size = round(span["size"], 1)
                    flags = span["flags"]   # bold=16, italic=2, etc.
                    size_counter[size] += 1
                    flag_counter[(size, flags)] += 1
                    if size not in samples:
                        samples[size] = []
                    if len(samples[size]) < 5:
                        samples[size].append({
                            "text": text[:80],
                            "flags": flags,
                            "bold": bool(flags & 16),
                            "italic": bool(flags & 2),
                            "page": page_num + 1
                        })

    print("FONT SIZE DISTRIBUTION (most common → least):")
    for size, count in sorted(size_counter.items(), key=lambda x: -x[1]):
        pct = count / sum(size_counter.values()) * 100
        bar = "█" * int(pct / 2)
        print(f"  Size {size:5.1f} | {count:4d}x ({pct:4.1f}%) {bar}")

    print("\nSAMPLES BY FONT SIZE (largest first):")
    for size in sorted(samples.keys(), reverse=True):
        print(f"\n  ── Size {size} ──")
        for s in samples[size]:
            bold_tag = "[BOLD]" if s["bold"] else "      "
            ital_tag = "[ITAL]" if s["italic"] else "      "
            print(f"    {bold_tag}{ital_tag} p{s['page']:02d}: {s['text']!r}")

    # Suggest heading threshold
    sizes = sorted(size_counter.keys(), reverse=True)
    body_size = sorted(size_counter.items(), key=lambda x: -x[1])[0][0]
    print(f"\n\nSUGGESTED STRATEGY:")
    print(f"  Body text size (most common): {body_size}")
    heading_sizes = [s for s in sizes if s > body_size]
    print(f"  Sizes larger than body (likely headings): {heading_sizes}")
    print(f"  Or: bold text at body size = sub-headings")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python pdf_diagnostic.py path/to/file.pdf")
    else:
        analyse_pdf(path)
