"""
FlashGenius — app.py
Pipeline on every upload:
  1. run_diagnostic(pdf)  → calibrated font/size profile (replaces pdf_diagnostic.py)
  2. extract_pdf_structure(pdf, profile) → headings with calibrated thresholds
  3. groq_cleanup_headings() → remove false positives, fix levels
  4. build_section_map() → heading → text slice
  Returns diagnostic summary to the dashboard for display.
"""

import os, json, re, traceback
from collections import Counter
from datetime import date, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import fitz          # PyMuPDF
import requests
from bs4 import BeautifulSoup
from groq import Groq

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client  = Groq(api_key=GROQ_API_KEY)

# Primary model + fallback chain (tried in order when quota/rate limit hit)
GROQ_PRIMARY   = "llama-3.3-70b-versatile"
GROQ_FALLBACKS = [
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama3-8b-8192",
]
GROQ_MODEL = GROQ_PRIMARY   # updated at runtime if primary fails


def groq_call(messages, system=None, temperature=0.3, max_tokens=4096):
    """Try primary model, fall back through list on quota/rate/model errors."""
    global GROQ_MODEL
    to_try = [GROQ_MODEL] + [m for m in GROQ_FALLBACKS if m != GROQ_MODEL]
    msgs   = ([{"role":"system","content":system}] if system else []) + list(messages)
    last   = None
    for model in to_try:
        try:
            resp = groq_client.chat.completions.create(
                model=model, messages=msgs,
                temperature=temperature, max_tokens=max_tokens,
            )
            if model != GROQ_MODEL:
                print(f"[Groq] Switched to fallback model: {model}")
                GROQ_MODEL = model
            return resp.choices[0].message.content.strip()
        except Exception as e:
            s = str(e).lower()
            if any(k in s for k in ('rate_limit','quota','model','tokens','429','503','decommissioned','not found')):
                print(f"[Groq] {model} unavailable ({type(e).__name__}), trying next…")
                last = e
                continue
            raise
    raise RuntimeError(f"All Groq models exhausted. Last: {last}")


# ════════════════════════════════════════════════════════════════════════════════
#  STEP 0 — PDF DIAGNOSTIC  (integrated; runs automatically on every upload)
#  Mirrors pdf_diagnostic.py logic — no need to run it separately.
# ════════════════════════════════════════════════════════════════════════════════

def run_diagnostic(filepath: str) -> dict:
    """
    Analyse font-size and bold distribution across all spans in the PDF.
    Returns a calibrated profile that drives heading extraction thresholds.
    Also prints the same output as pdf_diagnostic.py to the server console.
    """
    doc          = fitz.open(filepath)
    size_counter = Counter()
    bold_counter = Counter()     # size → bold span count
    samples      = {}            # size → list of sample text dicts

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text or len(text) < 2:
                        continue
                    sz    = round(span["size"], 1)
                    bold  = bool(span["flags"] & 16)
                    ital  = bool(span["flags"] & 2)
                    size_counter[sz] += 1
                    if bold:
                        bold_counter[sz] += 1
                    if sz not in samples:
                        samples[sz] = []
                    if len(samples[sz]) < 5:
                        samples[sz].append({
                            "text": text[:80], "bold": bold,
                            "italic": ital, "page": page_num + 1
                        })

    if not size_counter:
        return _default_profile()

    body_size     = size_counter.most_common(1)[0][0]
    all_sizes     = sorted(size_counter.keys(), reverse=True)
    # Only count sizes that appear meaningfully (>= 3 spans) as heading candidates
    heading_sizes = [s for s in all_sizes if s > body_size * 1.05 and size_counter[s] >= 3]
    bold_at_body  = bold_counter.get(body_size, 0) >= 2  # at least 2 bold spans at body size

    # ── Derive thresholds from actual size distribution ───────────
    # lvl1: the largest distinct size (chapter titles)
    # lvl2: anything meaningfully larger than body (sections)
    # lvl3: bold text at or near body size (sub-sections)
    if len(heading_sizes) >= 2:
        lvl1_min = heading_sizes[0] * 0.92          # top size with 8% tolerance
        lvl2_min = heading_sizes[-1] * 0.95         # smallest heading size
    elif len(heading_sizes) == 1:
        lvl1_min = heading_sizes[0] * 0.92
        lvl2_min = body_size * 1.04
    else:
        # No obvious heading sizes — rely on bold detection
        lvl1_min = body_size * 1.15
        lvl2_min = body_size * 1.04

    # Ensure lvl2 is always strictly above body
    lvl2_min = max(lvl2_min, body_size * 1.04)

    profile = {
        "body_size"    : body_size,
        "heading_sizes": heading_sizes,
        "size_counts"  : {str(k): v for k, v in size_counter.items()},
        "bold_at_body" : bold_at_body,
        "samples"      : {str(k): v for k, v in samples.items()},
        "suggested"    : {
            "lvl1_min"      : round(lvl1_min, 1),
            "lvl2_min"      : round(lvl2_min, 1),
            "lvl3_bold_only": bool(bold_at_body),
        }
    }

    # ── Console output (mirrors pdf_diagnostic.py) ────────────────
    total = sum(size_counter.values())
    print(f"\n{'='*62}")
    print(f" DIAGNOSTIC  {filepath}")
    print(f" Pages: {len(doc)}  |  Body size: {body_size}  |  Heading sizes: {heading_sizes}")
    print(f"{'='*62}")
    print(" FONT SIZE DISTRIBUTION (most common → least):")
    for sz, cnt in sorted(size_counter.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        bold_tag = f"[{bold_counter[sz]}bold]" if bold_counter.get(sz) else ""
        print(f"  {sz:5.1f} | {cnt:5d}x ({pct:4.1f}%)  {bar} {bold_tag}")
    print(f"\n SAMPLES BY SIZE (top sizes):")
    for sz in sorted(samples.keys(), reverse=True)[:6]:
        print(f"  ── size {sz} ──")
        for s in samples[sz]:
            tag = "[BOLD]" if s["bold"] else "      "
            print(f"    {tag} p{s['page']:02d}: {s['text']!r}")
    print(f"\n SUGGESTED THRESHOLDS:")
    print(f"  lvl1 (chapter)    ≥ {lvl1_min:.1f}")
    print(f"  lvl2 (section)    ≥ {lvl2_min:.1f}")
    print(f"  lvl3 (subsection) = bold @ body ({body_size}) → {bold_at_body}")
    print(f"{'='*62}\n")

    return profile


def _default_profile():
    return {
        "body_size": 10.0, "heading_sizes": [], "size_counts": {},
        "bold_at_body": True, "samples": {},
        "suggested": {"lvl1_min": 14.0, "lvl2_min": 12.0, "lvl3_bold_only": True}
    }


# ════════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PDF STRUCTURE EXTRACTION  (uses calibrated profile)
# ════════════════════════════════════════════════════════════════════════════════

NOISE = re.compile(
    r'^(\d+|[ivxlc]+\.?|page\s*\d+|figure\s*\d+|table\s*\d+|'
    r'©|www\.|http|all rights|printed in|isbn|published by|'
    r'exercise|activity|let us|do you know|think about)$',
    re.I
)

def extract_pdf_structure(filepath: str, profile: dict):
    doc  = fitz.open(filepath)
    sug  = profile.get("suggested", {})
    body = profile.get("body_size", 10.0)

    lvl1_min  = sug.get("lvl1_min",   body * 1.35)
    lvl2_min  = sug.get("lvl2_min",   body * 1.04)
    use_bold3 = sug.get("lvl3_bold_only", True)

    # Collect all lines with font metadata
    page_texts   = []
    headings_raw = []
    # Track repeated lines across pages (headers/footers repeat → skip them)
    line_page_count = Counter()   # line_text_lower → how many pages it appears on

    # First pass: count how many pages each line appears on
    for page_num, page in enumerate(doc):
        seen_this_page = set()
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"] != 0: continue
            for line in block["lines"]:
                lt = "".join(s["text"] for s in line["spans"]).strip()
                lk = lt.lower()
                if lt and lk not in seen_this_page:
                    seen_this_page.add(lk)
                    line_page_count[lk] += 1

    # Second pass: extract structure
    for page_num, page in enumerate(doc):
        page_text = ""
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"] != 0: continue
            for line in block["lines"]:
                line_text = "".join(s["text"] for s in line["spans"]).strip()
                if not line_text: continue
                page_text += line_text + "\n"

                sizes   = [s["size"] for s in line["spans"]]
                bolds   = [bool(s["flags"] & 16) for s in line["spans"]]
                dom_sz  = round(max(sizes), 1)
                # Bold if majority of character count is bold (more accurate than span count)
                total_chars = sum(len(s["text"]) for s in line["spans"])
                bold_chars  = sum(len(s["text"]) for s in line["spans"] if bool(s["flags"] & 16))
                is_bold = (bold_chars / max(total_chars, 1)) > 0.5

                headings_raw.append({
                    "text"   : line_text,
                    "size"   : dom_sz,
                    "bold"   : is_bold,
                    "page"   : page_num + 1,
                    "repeats": line_page_count.get(line_text.lower(), 1),
                })
        page_texts.append(page_text)

    full_text = "\n".join(page_texts)
    headings  = []
    seen_texts = set()

    for h in headings_raw:
        t       = h["text"].strip()
        sz      = h["size"]
        is_bold = h["bold"]
        repeats = h["repeats"]

        # Skip very short, very long, or repeated-across-pages lines (headers/footers)
        if not (3 <= len(t) <= 130): continue
        if repeats >= 3: continue          # appears on 3+ pages = header/footer
        if NOISE.match(t): continue
        if re.match(r'^\d+[\s\d.]*$', t): continue
        if not re.search(r'[A-Za-z]', t): continue
        # Skip lines that look like sentences (contain 5+ words with common body words)
        words = t.split()
        body_words = {'the','is','are','was','were','in','of','and','a','an','to','that',
                      'this','we','you','it','on','at','by','be','as','for','with','from'}
        if len(words) >= 7 and sum(1 for w in words if w.lower() in body_words) >= 3:
            continue

        key = t.lower()
        if key in seen_texts: continue

        # Assign heading level
        if sz >= lvl1_min:
            level = 1
        elif sz >= lvl2_min:
            level = 2
        elif use_bold3 and is_bold and sz >= body * 0.92:
            level = 3
        else:
            continue

        headings.append({"text": t, "level": level, "page": h["page"],
                         "font_size": sz, "bold": is_bold})
        seen_texts.add(key)

    headings = groq_cleanup_headings(headings, full_text[:3000])
    print(f"[PDF] body={body:.1f}  lvl1≥{lvl1_min:.1f}  lvl2≥{lvl2_min:.1f}  bold_l3={use_bold3}  → {len(headings)} headings")
    return full_text, headings, page_texts


# ════════════════════════════════════════════════════════════════════════════════
#  STEP 2 — GROQ HEADING CLEANUP
# ════════════════════════════════════════════════════════════════════════════════

def groq_cleanup_headings(raw_headings, text_sample):
    if not raw_headings:
        return []

    compact = [{"t": h["text"], "lv": h["level"], "pg": h["page"]} for h in raw_headings]
    prompt  = f"""You are reviewing heading candidates extracted from a textbook PDF via font-size analysis.

Your job:
1. Remove false positives: figure captions, table labels, publisher info, exercise numbers, page footers, random bold words that are not section titles
2. Keep all REAL chapter titles, section headings, sub-section headings, and topic names
3. Return in EXACT same page order
4. Clean up text (fix capitalisation, remove trailing colons if not needed)
5. Keep lv (1=chapter, 2=section, 3=subsection) and pg unchanged unless clearly wrong
6. Correct obvious level errors (e.g. a clear chapter title marked lv=3)

Output ONLY a JSON array. No markdown, no explanation.

CANDIDATE HEADINGS:
{json.dumps(compact, indent=2)}

Cleaned JSON array:"""

    try:
        raw = groq_call([{"role": "user", "content": prompt}], temperature=0.05, max_tokens=2048)
        raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\n?```$', '', raw, flags=re.MULTILINE)
        m   = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            cleaned = json.loads(m.group(0))
            result  = []
            for item in cleaned:
                t = str(item.get("t","")).strip()
                if t:
                    result.append({
                        "text"     : t,
                        "level"    : max(1, min(4, int(item.get("lv", 2)))),
                        "page"     : int(item.get("pg", 1)),
                        "font_size": 0,
                        "bold"     : True,
                    })
            print(f"[Groq cleanup] {len(raw_headings)} → {len(result)} headings")
            return result
    except Exception as e:
        print(f"[Groq cleanup error] {e}")
        traceback.print_exc()

    return raw_headings


# ════════════════════════════════════════════════════════════════════════════════
#  STEP 3 — SECTION MAP
# ════════════════════════════════════════════════════════════════════════════════

def build_section_map_from_headings(full_text, headings):
    if not headings:
        return {"Full Document": full_text[:8000]}

    text_lower = full_text.lower()
    positions  = []

    for h in headings:
        t   = h["text"]
        idx = full_text.find(t)
        if idx == -1:
            idx = text_lower.find(t.lower())
        if idx == -1:
            for w in sorted(t.split(), key=len, reverse=True):
                if len(w) > 4:
                    idx = text_lower.find(w.lower())
                    if idx != -1:
                        break
        positions.append((t, max(idx, 0), h["level"]))

    positions.sort(key=lambda x: x[1])
    section_map = {}

    for i, (title, pos, level) in enumerate(positions):
        end   = positions[i+1][1] if i+1 < len(positions) else len(full_text)
        block = full_text[pos:end].strip()
        if len(block) < 100 and i > 0:
            block = full_text[positions[i-1][1]:end].strip()
        section_map[title] = block[:7000]

    return section_map


# ════════════════════════════════════════════════════════════════════════════════
#  WEB SUPPLEMENT
# ════════════════════════════════════════════════════════════════════════════════

def web_search_topic(topic):
    try:
        clean = re.sub(r'^\d+[\.\d]*\s*', '', topic).strip()
        url   = f"https://en.wikipedia.org/wiki/{clean.replace(' ','_')}"
        r     = requests.get(url, headers={'User-Agent': 'FlashGeniusBot/2.0'}, timeout=7)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            for t in soup(['script','style','table','sup','cite']):
                t.decompose()
            text = ' '.join(p.get_text() for p in soup.find_all('p')[:12])
            if len(text) > 100:
                return text[:3000]
    except Exception:
        pass
    return ""


# ════════════════════════════════════════════════════════════════════════════════
#  FLASHCARD GENERATION  — concept extraction → card generation
# ════════════════════════════════════════════════════════════════════════════════

CONCEPT_SYSTEM = """You are an elite educator and memory scientist.

Analyse the given educational content and extract EVERY important concept needed for long-term retention.

For each concept identify:
- concept_name  : short label (3-8 words)
- category      : one of [definition, mechanism, formula, application, example, comparison, misconception, process, timeline]
- core_idea     : one sentence capturing the essential insight
- sub_points    : list of key supporting details (max 5)
- related       : other concept names this connects to
- common_errors : 1-2 mistakes students typically make (empty list if none)

Cover ALL of:
✓ Every defined term
✓ Every process/mechanism (how/why)
✓ Every formula, equation, or rule
✓ Every real-world application or example
✓ Every comparison between things
✓ Every common misconception or edge case
✓ Cause-effect relationships
✓ Exceptions to rules

Output ONLY a valid JSON array. No markdown, no preamble.

[
  {
    "concept_name": "...",
    "category": "...",
    "core_idea": "...",
    "sub_points": ["..."],
    "related": ["..."],
    "common_errors": ["..."]
  }
]"""

CARD_SYSTEM = """You are an elite educator and memory scientist.
Your goal is to MAXIMISE long-term retention using active recall and spaced repetition principles.

Generate flashcards for the given concept. Use EVERY applicable card type:

  DEFINITION    → "What is X?" / "Define X"
  CONCEPT       → "How does X work?" / "Why does X happen?"
  FORMULA       → Recall of equation, rule, or symbolic form
  APPLICATION   → Real-world or numerical problem using the concept
  EXAMPLE       → Identify/explain a concrete case or scenario
  COMPARE       → How X differs from or relates to Y
  ACTIVE_RECALL → Fill-in-blank, complete-the-process, or sequence question
  ERROR_BASED   → "A student claims X. What is wrong?" — targets misconceptions

QUALITY BAR:
  ✓ Fronts: genuine exam-style questions, not statements
  ✓ Backs: complete + scannable; use \\n• for bullet points
  ✓ Formula/application cards must include worked steps
  ✓ ERROR_BASED cards: state the wrong belief AND the correct one
  ✓ DO NOT limit card count — generate as many as the concept needs

Output ONLY a valid JSON array. No markdown, no text before [ or after ].

[
  {
    "front": "...",
    "back": "...",
    "type": "DEFINITION|CONCEPT|FORMULA|APPLICATION|EXAMPLE|COMPARE|ACTIVE_RECALL|ERROR_BASED",
    "difficulty": "easy|medium|hard",
    "hint": "...",
    "related_to": ["..."],
    "ease_factor": 2.5,
    "interval": 1,
    "repetitions": 0,
    "next_review": null
  }
]"""

TYPE_GUIDANCE = {
    'definition':    'Must include: DEFINITION, CONCEPT, ACTIVE_RECALL. Add EXAMPLE if possible.',
    'mechanism':     'Must include: CONCEPT, ACTIVE_RECALL. Add APPLICATION or EXAMPLE.',
    'formula':       'Must include: FORMULA, APPLICATION (with worked steps), ACTIVE_RECALL. Add ERROR_BASED if errors exist.',
    'application':   'Must include: APPLICATION, EXAMPLE, ACTIVE_RECALL.',
    'example':       'Must include: EXAMPLE, ACTIVE_RECALL. Add COMPARE if related concepts exist.',
    'comparison':    'Must include: COMPARE (at least 2), ACTIVE_RECALL, DEFINITION for each item.',
    'misconception': 'Must include: ERROR_BASED (at least 2), CONCEPT. Wrong belief → correct one.',
    'process':       'Must include: CONCEPT (step-by-step), ACTIVE_RECALL (sequence), APPLICATION.',
    'timeline':      'Must include: ACTIVE_RECALL (order), EXAMPLE, APPLICATION.',
}


def extract_concepts(topic, doc_text, web_text):
    user_msg = f"""TOPIC: {topic}

=== TEXTBOOK CONTENT ===
{doc_text[:4500]}

=== SUPPLEMENTARY WEB CONTENT ===
{web_text[:1500]}

Extract ALL concepts from the above content. Output ONLY the JSON array."""

    raw = groq_call([{"role":"user","content":user_msg}], system=CONCEPT_SYSTEM, temperature=0.2, max_tokens=4096)
    raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\n?```$', '', raw, flags=re.MULTILINE)
    m   = re.search(r'\[.*\]', raw, re.DOTALL)
    if not m:
        raise ValueError(f"No concept JSON. Raw: {raw[:200]}")
    concepts = json.loads(m.group(0))
    print(f"  [Concepts] {len(concepts)} extracted")
    return concepts


def generate_cards_for_concept(concept, topic):
    cat      = concept.get('category', 'definition')
    guidance = TYPE_GUIDANCE.get(cat, 'Include DEFINITION, CONCEPT, and ACTIVE_RECALL.')

    user_msg = f"""TOPIC: {topic}
CONCEPT: {concept['concept_name']}
CATEGORY: {cat}
CORE IDEA: {concept['core_idea']}
KEY DETAILS: {json.dumps(concept.get('sub_points', []))}
RELATED CONCEPTS: {json.dumps(concept.get('related', []))}
COMMON STUDENT ERRORS: {json.dumps(concept.get('common_errors', []))}

Card type guidance → {guidance}

Generate ALL applicable card types. No card limit. Output ONLY the JSON array."""

    raw = groq_call([{"role":"user","content":user_msg}], system=CARD_SYSTEM, temperature=0.5, max_tokens=3000)
    raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\n?```$', '', raw, flags=re.MULTILINE)
    m   = re.search(r'\[.*\]', raw, re.DOTALL)
    if not m:
        raise ValueError(f"No card JSON: {raw[:200]}")
    cards = json.loads(m.group(0))

    for c in cards:
        c['topic']       = topic
        c['concept']     = concept['concept_name']
        c['category']    = cat
        c['related_to']  = c.get('related_to', concept.get('related', []))
        c['ease_factor'] = float(c.get('ease_factor', 2.5))
        c['interval']    = int(c.get('interval', 1))
        c['repetitions'] = int(c.get('repetitions', 0))
        c['next_review'] = c.get('next_review', None)
        if c.get('difficulty') not in ('easy', 'medium', 'hard'):
            c['difficulty'] = 'medium'
        if 'hint' not in c: c['hint'] = ''
        if 'type' not in c: c['type'] = 'CONCEPT'
    return cards


def generate_flashcards_ai(topic, doc_text, web_text):
    """
    Single-pass generation: extract concepts + generate all cards in ONE Groq call.
    Falls back to direct card generation if concept extraction fails.
    """
    # ── Attempt 1: full two-pass pipeline ────────────────────────
    try:
        concepts = extract_concepts(topic, doc_text, web_text)
        if concepts:
            all_cards, seen_fronts = [], set()
            for i, concept in enumerate(concepts):
                try:
                    cards = generate_cards_for_concept(concept, topic)
                    for c in cards:
                        key = re.sub(r'\s+', ' ', c.get('front', '')).lower().strip()[:80]
                        if key not in seen_fronts:
                            seen_fronts.add(key)
                            all_cards.append(c)
                    print(f"  [Cards] {i+1}/{len(concepts)} '{concept['concept_name']}' → {len(cards)} cards")
                except Exception as e:
                    print(f"  [Cards] ⚠ Skipped concept: {e}")
            if all_cards:
                for idx, c in enumerate(all_cards):
                    c['id'] = idx + 1
                return all_cards
    except Exception as e:
        print(f"  [Two-pass] failed ({e}), falling back to single-pass...")

    # ── Fallback: single direct call ─────────────────────────────
    return _generate_cards_direct(topic, doc_text, web_text)


DIRECT_SYSTEM = """You are an expert educator. Generate comprehensive flashcards for the given topic.

Use these card types appropriately: DEFINITION, CONCEPT, FORMULA, APPLICATION, EXAMPLE, COMPARE, ACTIVE_RECALL, ERROR_BASED

Rules:
- Generate 10-20 cards covering ALL important aspects of the topic
- Fronts: clear exam-style questions
- Backs: complete answers; use \\n• for bullet lists
- Mix card types and difficulty levels
- Output ONLY a valid JSON array, nothing else

[
  {
    "front": "question text",
    "back": "answer text",
    "type": "DEFINITION",
    "difficulty": "easy",
    "hint": "optional hint",
    "concept": "sub-topic name",
    "related_to": [],
    "ease_factor": 2.5,
    "interval": 1,
    "repetitions": 0,
    "next_review": null
  }
]"""


def _generate_cards_direct(topic, doc_text, web_text):
    """Single Groq call — more reliable fallback."""
    prompt = f"""TOPIC: {topic}

TEXTBOOK CONTENT:
{doc_text[:5000]}

WEB CONTENT:
{web_text[:1500]}

Generate 10-20 comprehensive flashcards covering ALL concepts in this topic.
Output ONLY the JSON array."""

    raw = groq_call([{"role":"user","content":prompt}], system=DIRECT_SYSTEM, temperature=0.4, max_tokens=4096)
    raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\n?```$', '', raw, flags=re.MULTILINE)
    m   = re.search(r'\[.*\]', raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array in direct generation response. Raw: {raw[:300]}")
    cards = json.loads(m.group(0))
    for idx, c in enumerate(cards):
        c['id']          = idx + 1
        c['topic']       = topic
        c['concept']     = c.get('concept', topic)
        c['ease_factor'] = float(c.get('ease_factor', 2.5))
        c['interval']    = int(c.get('interval', 1))
        c['repetitions'] = int(c.get('repetitions', 0))
        c['next_review'] = c.get('next_review', None)
        c['related_to']  = c.get('related_to', [])
        if c.get('difficulty') not in ('easy','medium','hard'):
            c['difficulty'] = 'medium'
        if 'hint' not in c: c['hint'] = ''
        if 'type' not in c: c['type'] = 'CONCEPT'
    print(f"  [Direct] Generated {len(cards)} cards for: {topic}")
    return cards


# ════════════════════════════════════════════════════════════════════════════════
#  SM-2 SPACED REPETITION
# ════════════════════════════════════════════════════════════════════════════════

def sm2_update(card, quality):
    ef = card.get('ease_factor', 2.5)
    n  = card.get('repetitions', 0)
    iv = card.get('interval', 1)
    if quality >= 3:
        iv = 1 if n == 0 else (6 if n == 1 else round(iv * ef))
        n += 1
    else:
        n, iv = 0, 1
    ef = max(1.3, ef + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return {**card,
            'ease_factor': round(ef, 2),
            'interval': iv,
            'repetitions': n,
            'next_review': (date.today() + timedelta(days=iv)).isoformat()}


def _fuzzy_find(full_text, topic):
    lines  = full_text.split('\n')
    tc     = re.sub(r'^\d+[\.\d]*\s*', '', topic).strip().lower()
    twords = set(w for w in tc.split() if len(w) > 2)
    if not twords:
        return full_text[:4000]
    best_i, best_sc = 0, 0
    for i, line in enumerate(lines):
        sc = len(twords & set(line.lower().split())) * 2 + (10 if tc in line.lower() else 0)
        if sc > best_sc:
            best_sc, best_i = sc, i
    return '\n'.join(lines[max(0, best_i-3):min(len(lines), best_i+200)])


# ════════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/viewer')
def viewer():
    return send_from_directory('static', 'viewer.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    save_path = os.path.join(UPLOAD_FOLDER, 'current.pdf')
    file.save(save_path)

    # ── Step 0: Diagnostic ─────────────────────────────────────────
    try:
        profile = run_diagnostic(save_path)
    except Exception as e:
        print(f"[Diagnostic error] {e} — using defaults")
        profile = _default_profile()

    # ── Step 1: Structure extraction ───────────────────────────────
    try:
        full_text, headings, page_texts = extract_pdf_structure(save_path, profile)
    except Exception as e:
        return jsonify({'error': f'PDF parse error: {e}'}), 500

    with open(os.path.join(UPLOAD_FOLDER, 'text.txt'), 'w', encoding='utf-8') as f:
        f.write(full_text)

    section_map = build_section_map_from_headings(full_text, headings)
    sections    = [h["text"] for h in headings]

    with open(os.path.join(UPLOAD_FOLDER, 'sections.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'sections'   : sections,
            'headings'   : headings,
            'section_map': section_map,
            'profile'    : profile,
        }, f, ensure_ascii=False)

    # Compact diagnostic summary for the dashboard
    diag = {
        "body_size"     : profile["body_size"],
        "heading_sizes" : profile["heading_sizes"][:6],
        "lvl1_min"      : profile["suggested"]["lvl1_min"],
        "lvl2_min"      : profile["suggested"]["lvl2_min"],
        "bold_subheads" : profile["bold_at_body"],
        "total_headings": len(headings),
        "pages"         : len(page_texts),
    }

    print(f"[Upload] {len(page_texts)} pages, {len(sections)} sections detected")
    return jsonify({
        'sections'  : sections,
        'headings'  : headings,
        'pages'     : len(page_texts),
        'diagnostic': diag,
    })


@app.route('/generate', methods=['POST'])
def generate():
    data   = request.json
    topics = data.get('topics', [])
    if not topics:
        return jsonify({'error': 'No topics provided'}), 400

    text_path     = os.path.join(UPLOAD_FOLDER, 'text.txt')
    sections_path = os.path.join(UPLOAD_FOLDER, 'sections.json')
    if not os.path.exists(text_path):
        return jsonify({'error': 'No document uploaded. Please upload a PDF first.'}), 400

    with open(text_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    section_map = {}
    if os.path.exists(sections_path):
        with open(sections_path, 'r', encoding='utf-8') as f:
            section_map = json.load(f).get('section_map', {})

    all_cards, errors = {}, {}

    for topic in topics:
        print(f"\n[Generate] ── {topic}")
        doc_chunk = section_map.get(topic, '')
        if not doc_chunk or len(doc_chunk.strip()) < 80:
            doc_chunk = _fuzzy_find(full_text, topic)
        if len(doc_chunk.strip()) < 30:
            doc_chunk = full_text[:5000]

        web_text = web_search_topic(topic)
        try:
            cards = generate_flashcards_ai(topic, doc_chunk, web_text)
            if cards:
                all_cards[topic] = cards
                print(f"[Generate] ✅ {len(cards)} cards for: {topic}")
            else:
                errors[topic] = "Empty response from AI"
                all_cards[topic] = []
                print(f"[Generate] ⚠ 0 cards for: {topic}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Generate] ❌ {topic}:\n{tb}")
            errors[topic]    = str(e)
            all_cards[topic] = []

    # Only include topics that actually got cards in the topics list
    successful_topics = [t for t in topics if all_cards.get(t)]

    # If nothing worked at all, return a clear error
    if not successful_topics:
        error_detail = "; ".join(f"{t}: {e}" for t, e in errors.items())
        return jsonify({
            'error': f'No flashcards could be generated. Details: {error_detail}',
            'flashcards': all_cards,
            'topics': list(all_cards.keys()),
            'errors': errors,
        }), 500

    result = {
        'flashcards': all_cards,
        'topics': successful_topics,  # only non-empty topics drive viewer sidebar
    }
    if errors:
        result['errors'] = errors
        result['partial'] = True
    return jsonify(result)


@app.route('/review', methods=['POST'])
def review():
    data    = request.json
    updated = sm2_update(data.get('card', {}), int(data.get('quality', 3)))
    return jsonify({'card': updated})


if __name__ == '__main__':
    if not GROQ_API_KEY:
        print("\n⚠️  GROQ_API_KEY not set!\n")
    print("🚀 FlashGenius → http://localhost:5050\n")
    app.run(debug=True, port=5050, threaded=True)
