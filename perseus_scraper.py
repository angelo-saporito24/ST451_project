"""
Perseus Digital Library Scraper
================================
Retrieves Greek + English text from the Scaife Viewer JSON API for
Plato and Aristotle, with cross-lingual phrase alignment via LaBSE
(Jaccard offset fallback if sentence-transformers is not installed).

Supported authors and works
----------------------------
Plato      : ion, phaedrus, republic, symposium, sophist
             (plus full PLATO_WORKS table for corpus scraping)
Aristotle  : nicomachean ethics, metaphysics, poetics
             Section keys are book/chapter numbers ("1", "2", …)
             NOT Bekker numbers — those are milestone divs in the HTML.

Usage
-----
    scraper = PerseusScraper()

    # Single passage
    p = scraper.get_passage("phaedrus", "243", author="plato")
    p = scraper.get_passage("nicomachean ethics", "1", author="aristotle")

    # Range
    p = scraper.get_passage_range("phaedrus", "243", "245", author="plato")

    # Alignment
    hit = p.find_greek("a god or something divine")
    print(hit)

Dependencies
------------
    pip install requests beautifulsoup4 lxml
    pip install sentence-transformers   # optional: LaBSE alignment
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _LABSE_AVAILABLE = True
except ImportError:
    _LABSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# TLG lookup tables
# ---------------------------------------------------------------------------

AUTHOR_IDS = {
    "plato":     "tlg0059",
    "aristotle": "tlg0086",
}

PLATO_WORKS = {
    # IDs confirmed from Scaife TLG index (tlg0059.tlgXXX)
    "euthyphro":   "tlg001",
    "apology":     "tlg002",
    "crito":       "tlg003",
    "phaedo":      "tlg004",
    "cratylus":    "tlg005",
    "theaetetus":  "tlg006",
    "sophist":     "tlg007",
    "statesman":   "tlg008",
    "parmenides":  "tlg009",
    "philebus":    "tlg010",
    "symposium":   "tlg011",
    "phaedrus":    "tlg012",
    "charmides":   "tlg018",
    "laches":      "tlg019",
    "lysis":       "tlg020",
    "euthydemus":  "tlg021",
    "protagoras":  "tlg022",
    "gorgias":     "tlg023",
    "meno":        "tlg024",
    "ion":         "tlg027",
    "republic":    "tlg030",
    "timaeus":     "tlg031",
    "critias":     "tlg032",
    "laws":        "tlg034",
}

ARISTOTLE_WORKS = {
    # IDs confirmed from Scaife TLG index (tlg0086.tlgXXX)
    "de anima":            "tlg002",
    "nicomachean ethics":  "tlg010",
    "metaphysics":         "tlg025",
    "politics":            "tlg035",
    "poetics":             "tlg034",
    "rhetoric":            "tlg038",
}

# CTS section counts — used by get_all_sections() to enumerate full works.
# Aristotle: book/chapter numbers ("1", "2", …), NOT Bekker numbers.
# Plato: Stephanus page numbers (first and last page of each work).
ARISTOTLE_SECTION_COUNTS = {
    "nicomachean ethics": 10,
    "metaphysics":        14,
    "poetics":            26,
    "politics":           8,
    "rhetoric":           3,
}

PLATO_STEPHANUS_RANGES = {
    "ion":        (530, 542),
    "phaedrus":   (227, 279),
    "republic":   (327, 621),
    "sophist":    (216, 268),
    "symposium":  (172, 223),
    "apology":    (17,  42),
    "meno":       (70,  100),
    "phaedo":     (57,  118),
    "timaeus":    (17,  92),
    "parmenides": (126, 166),
    "theaetetus": (142, 210),
    "philebus":   (11,  67),
    "laws":       (624, 969),
    "gorgias":    (447, 527),
    "protagoras": (309, 362),
}

WORK_TABLES = {
    "plato":     PLATO_WORKS,
    "aristotle": ARISTOTLE_WORKS,
}

GREEK_EDITIONS   = ["perseus-grc2", "perseus-grc1"]
ENGLISH_EDITIONS = ["perseus-eng2", "perseus-eng1"]


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def parse_reference(ref: str) -> tuple[str, Optional[str]]:
    """Split "243b" → ("243", "b");  "243" → ("243", None)."""
    m = re.fullmatch(r"(\d+)([a-e])?", ref.strip())
    return (m.group(1), m.group(2)) if m else (ref.strip(), None)


def stephanus_range(start: str, end: str) -> list[str]:
    """Integer Stephanus sections from start to end inclusive."""
    s = int(re.match(r"\d+", start).group())
    e = int(re.match(r"\d+", end).group())
    return [str(n) for n in range(s, e + 1)]


def get_all_sections(author: str, work: str) -> list[str]:
    """
    Return the full list of valid top-level CTS section keys for a work.

    Aristotle: book/chapter numbers ["1", "2", …] from known counts.
    Plato:     Stephanus page numbers ["227", "228", …] from known ranges.
    Both are hardcoded to avoid rate-limit failures on the Scaife
    children endpoint and Bekker number confusion for Aristotle.
    """
    author = author.lower()
    work   = work.lower()

    if author == "aristotle":
        n = ARISTOTLE_SECTION_COUNTS.get(work, 10)
        return [str(i) for i in range(1, n + 1)]

    if author == "plato":
        # Some works are structured by book at the top CTS level,
        # not by Stephanus page. These need book numbers, not page ranges.
        PLATO_BOOK_COUNTS = {
            "republic": 10,
            "laws":     12,
            "timaeus":  1,   # single book
        }
        if work in PLATO_BOOK_COUNTS:
            n = PLATO_BOOK_COUNTS[work]
            return [str(i) for i in range(1, n + 1)]
        rng = PLATO_STEPHANUS_RANGES.get(work)
        if rng:
            return [str(n) for n in range(rng[0], rng[1] + 1)]

    return []


# ---------------------------------------------------------------------------
# Alignment result
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    query:          str
    english_tokens: list[str]
    greek_tokens:   list[str]
    english_text:   str
    greek_text:     str
    confidence:     float
    method:         str

    def __str__(self):
        bar = "─" * 52
        return (
            f"\n{bar}\n"
            f"  Query     : \"{self.query}\"\n"
            f"  Method    : {self.method}\n"
            f"  Confidence: {self.confidence:.0%}\n"
            f"{bar}\n"
            f"  English → {self.english_text}\n"
            f"  Greek   → {self.greek_text}\n"
            f"{bar}\n"
        )


# ---------------------------------------------------------------------------
# Passage
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    author:         str
    work:           str
    reference:      str
    urn_greek:      str
    urn_english:    str
    greek_text:     Optional[str]
    english_text:   Optional[str]
    greek_tokens:   list[dict] = field(default_factory=list)
    english_tokens: list[dict] = field(default_factory=list)
    _greek_sents:   list[str]  = field(default_factory=list, repr=False)
    _english_sents: list[str]  = field(default_factory=list, repr=False)
    _labse_model:   object     = field(default=None,         repr=False)

    def __str__(self):
        bar = "─" * 60
        return (
            f"\n{bar}\n"
            f"  {self.author.title()} · {self.work.title()} · {self.reference}\n"
            f"{bar}\n\n"
            f"── GREEK ──────────────────────────────────────────────────\n\n"
            f"{self.greek_text or '[not retrieved]'}\n\n"
            f"── ENGLISH ─────────────────────────────────────────────────\n\n"
            f"{self.english_text or '[not retrieved]'}\n\n"
            f"{bar}\n"
        )

    def find_greek(self, english_phrase: str,
                   window_tokens: int = 3) -> Optional[AlignmentResult]:
        if _LABSE_AVAILABLE:
            return self._align_labse(english_phrase, window_tokens)
        return self._align_offset(english_phrase, window_tokens)

    def _align_labse(self, phrase: str, window: int) -> Optional[AlignmentResult]:
        if self._labse_model is None:
            print("[LaBSE] Loading model (first call only)…")
            self._labse_model = SentenceTransformer("LaBSE")
        if not self._greek_sents:
            self._greek_sents   = _chunk_sentences(self.greek_text   or "")
            self._english_sents = _chunk_sentences(self.english_text or "")
        if not self._greek_sents or not self._english_sents:
            return self._align_offset(phrase, window)

        model      = self._labse_model
        all_texts  = [phrase] + self._greek_sents
        embeddings = model.encode(all_texts, normalize_embeddings=True)
        q_emb      = embeddings[0]
        g_embs     = embeddings[1:]
        sims       = g_embs @ q_emb
        best_idx   = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_grc   = self._greek_sents[best_idx]

        e_embs   = model.encode(self._english_sents, normalize_embeddings=True)
        e_sims   = e_embs @ q_emb
        best_eng = self._english_sents[int(np.argmax(e_sims))]

        return AlignmentResult(
            query=phrase,
            english_tokens=best_eng.split(),
            greek_tokens=best_grc.split(),
            english_text=best_eng,
            greek_text=best_grc,
            confidence=best_score,
            method=f"LaBSE cosine (Greek sentence {best_idx})",
        )

    def _align_offset(self, phrase: str, window: int) -> Optional[AlignmentResult]:
        if not self.english_tokens or not self.greek_tokens:
            print("Token lists not populated.")
            return None
        eng_words   = [t["w"] for t in self.english_tokens]
        grc_words   = [t["w"] for t in self.greek_tokens]
        query_words = phrase.strip().split()

        best_start, best_len, best_score = _best_span_match(query_words, eng_words)
        if best_score == 0:
            return None

        best_end   = best_start + best_len
        n_eng, n_grc = len(eng_words), len(grc_words)
        frac_start = best_start / n_eng
        frac_end   = best_end   / n_eng
        grc_center = int((frac_start + frac_end) / 2 * n_grc)
        half_span  = max(1, int((frac_end - frac_start) * n_grc / 2))
        grc_start  = max(0,     grc_center - half_span - window)
        grc_end    = min(n_grc, grc_center + half_span + window + 1)

        return AlignmentResult(
            query=phrase,
            english_tokens=eng_words[best_start:best_end],
            greek_tokens=grc_words[grc_start:grc_end],
            english_text=" ".join(eng_words[best_start:best_end]),
            greek_text=" ".join(grc_words[grc_start:grc_end]),
            confidence=best_score,
            method=f"Jaccard offset (eng {best_start}:{best_end} → grc {grc_start}:{grc_end})",
        )


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

class PerseusScraper:

    SCAIFE_JSON = "https://scaife.perseus.org/library/passage/{urn}/json/"
    HEADERS = {
        "User-Agent": "PerseusScraper/1.0 (academic research)",
        "Accept":     "application/json",
    }

    def __init__(self, delay: float = 1.5):
        self.delay   = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def get_passage(self, work: str, reference: str,
                    author: str = "plato") -> Passage:
        section, sub = parse_reference(reference)
        return self._fetch_sections(work, author, [section], sub)

    def get_passage_range(self, work: str, start: str, end: str,
                          author: str = "plato") -> Passage:
        sections = stephanus_range(start, end)
        return self._fetch_sections(work, author, sections,
                                    sub=None, ref_label=f"{start}–{end}")

    def _fetch_sections(self, work: str, author: str, sections: list[str],
                        sub: Optional[str],
                        ref_label: Optional[str] = None) -> Passage:
        author_key = author.lower()
        work_key   = work.lower()
        author_id  = AUTHOR_IDS.get(author_key)
        if not author_id:
            raise ValueError(f"Unknown author '{author}'. "
                             f"Supported: {list(AUTHOR_IDS)}")
        work_id = WORK_TABLES.get(author_key, {}).get(work_key)
        if not work_id:
            raise ValueError(f"Unknown work '{work}' for {author}. "
                             f"Supported: {list(WORK_TABLES[author_key])}")

        urn_base = f"urn:cts:greekLit:{author_id}.{work_id}"
        grc_texts, eng_texts   = [], []
        grc_toks_all, eng_toks_all = [], []

        for i, sec in enumerate(sections):
            active_sub = sub if len(sections) == 1 else None

            grc_data = self._fetch_raw(urn_base, sec, GREEK_EDITIONS)
            if grc_data:
                t = _parse_html_to_text(grc_data["text_html"],
                                        start_milestone=(f"{sec}{active_sub}"
                                                         if active_sub else None))
                if t:
                    grc_texts.append(t)
                grc_toks_all.extend(_filter_word_tokens(
                    grc_data.get("word_tokens", [])))

            time.sleep(self.delay)

            eng_data = self._fetch_raw(urn_base, sec, ENGLISH_EDITIONS)
            if eng_data:
                t = _parse_html_to_text(eng_data["text_html"],
                                        start_milestone=(f"{sec}{active_sub}"
                                                         if active_sub else None))
                if t:
                    eng_texts.append(t)
                eng_toks_all.extend(_filter_word_tokens(
                    eng_data.get("word_tokens", [])))

            if i < len(sections) - 1:
                time.sleep(self.delay)

        ref = ref_label or (f"{sections[0]}{sub}" if sub else sections[0])
        return Passage(
            author=author, work=work, reference=ref,
            urn_greek=f"{urn_base}.{GREEK_EDITIONS[0]}:{sections[0]}",
            urn_english=f"{urn_base}.{ENGLISH_EDITIONS[0]}:{sections[0]}",
            greek_text="\n\n".join(grc_texts) or None,
            english_text="\n\n".join(eng_texts) or None,
            greek_tokens=grc_toks_all,
            english_tokens=eng_toks_all,
        )

    def _fetch_raw(self, urn_base: str, section: str,
                   editions: list[str]) -> Optional[dict]:
        for edition in editions:
            urn = f"{urn_base}.{edition}:{section}"
            url = self.SCAIFE_JSON.format(urn=urn)
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("text_html"):
                        return data
                    print(f"[Scaife] 200 but no text_html for {urn}")
                else:
                    print(f"[Scaife] HTTP {resp.status_code} for {urn}")
            except (requests.RequestException, ValueError) as e:
                print(f"[Scaife] Error: {e}")
            time.sleep(self.delay)
        return None


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _parse_html_to_text(html: str,
                        start_milestone: Optional[str] = None) -> Optional[str]:
    """
    Parse Scaife text_html into plain text.

    Format A — Plato dialogues: <div class="said"> blocks with
               <span class="label"> speaker names.
    Format B — Aristotle prose: no "said" divs, bare <t> tokens
               throughout the document.
    """
    soup      = BeautifulSoup(html, "lxml")
    said_divs = soup.find_all("div", class_="said")
    if said_divs:
        return _parse_dialogue(soup, said_divs, start_milestone)
    return _parse_prose(soup, start_milestone)


def _parse_dialogue(soup, said_divs, start_milestone):
    collecting = (start_milestone is None)
    done       = False
    parts      = []

    for said in said_divs:
        if done:
            break
        label   = said.find("span", class_="label")
        speaker = _extract_speaker(label)
        tokens  = []

        for elem in said.descendants:
            if not hasattr(elem, "name"):
                continue
            if label and _is_inside(elem, label):
                continue
            if elem.name == "div" and "milestone" in elem.get("class", []):
                ms = elem.get_text().strip()
                if start_milestone and ms == start_milestone:
                    collecting = True
                    tokens     = []
                elif collecting and start_milestone:
                    done = True
                    break
                continue
            if elem.name == "t" and collecting:
                tokens.append({"w": elem.get("w", ""), "t": elem.get("t", "w")})

        if tokens:
            parts.append((speaker, tokens))

    if not parts:
        return None
    lines = []
    for speaker, tokens in parts:
        text = _tokens_to_text(tokens)
        if text:
            lines.append(f"{speaker}.\n{text}" if speaker else text)
    return "\n\n".join(lines).strip() or None


def _parse_prose(soup, start_milestone):
    collecting = (start_milestone is None)
    tokens     = []

    for elem in soup.descendants:
        if not hasattr(elem, "name"):
            continue
        if elem.name == "div" and "milestone" in elem.get("class", []):
            ms = elem.get_text().strip()
            if start_milestone and ms == start_milestone:
                collecting = True
                tokens     = []
            elif collecting and start_milestone and tokens:
                break
            continue
        if elem.name == "t" and collecting:
            tokens.append({"w": elem.get("w", ""), "t": elem.get("t", "w")})

    if not tokens:
        return None
    return _tokens_to_text(tokens) or None


def _extract_speaker(label_span) -> str:
    if not label_span:
        return ""
    return "".join(t.get("w", "") for t in label_span.find_all("t")
                   if t.get("t", "w") == "w")


def _is_inside(elem, ancestor) -> bool:
    return ancestor in elem.parents


def _tokens_to_text(tokens: list[dict]) -> str:
    parts, prev_word = [], False
    for tok in tokens:
        w, t = tok.get("w", ""), tok.get("t", "w")
        if not w:
            continue
        if t == "p":
            parts.append(w)
            prev_word = False
        else:
            if parts and prev_word:
                parts.append(" ")
            parts.append(w)
            prev_word = True
    return "".join(parts).strip()


def _filter_word_tokens(tokens: list[dict]) -> list[dict]:
    seen = set()
    out  = []
    for t in tokens:
        if t.get("t") == "w" and t.get("w"):
            key = (t["w"], t.get("o", 0))
            if key not in seen:
                seen.add(key)
                out.append({"w": t["w"], "o": t.get("o", 0)})
    return out


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _chunk_sentences(text: str) -> list[str]:
    chunks = []
    for turn in text.split("\n\n"):
        lines = turn.strip().splitlines()
        body  = " ".join(l for l in lines
                         if not re.match(r"^[A-ZΑ-Ω]{1,10}\.$", l.strip()))
        for sent in re.split(r"(?<=[.;?·])\s+", body):
            sent = sent.strip()
            if len(sent.split()) >= 3:
                chunks.append(sent)
    return chunks


def _best_span_match(query: list[str], tokens: list[str]
                     ) -> tuple[int, int, float]:
    q_set = {w.lower() for w in query}
    q_len = len(query)
    n     = len(tokens)
    best  = (0, q_len, 0.0)
    for span_len in range(max(1, q_len - 1), q_len + 2):
        for start in range(n - span_len + 1):
            score = _jaccard(q_set,
                             {w.lower() for w in tokens[start:start + span_len]})
            if score > best[2]:
                best = (start, span_len, score)
    return best


def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0