"""
Token Filtering — Stage 1 of the LDA Preprocessing Pipeline
=============================================================
Cleans raw token lists from the Perseus scraper JSON files before
morphological analysis and DTM construction.

Removes:
  - Speaker labels      (ΣΩ, ΙΩΝ, ΚΡ, ΑΠΟΛ, ΕΤ, …)
  - Perseus citations   (Hom, Il, Od, ff, and bare numerics)
  - Punctuation tokens
  - Non-Greek tokens    (anything without Unicode Greek characters)
  - Very short tokens   (fewer than MIN_CHARS characters after normalisation)

Usage:
    from token_filter import filter_tokens, filter_corpus

    # Single document
    clean = filter_tokens(doc["tokens"])

    # Full corpus dict (output of pipeline.scrape_corpus)
    clean_corpus = filter_corpus(corpus)
"""

import re
import json
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CHARS = 2   # drop tokens shorter than this after normalisation

# Unicode ranges that contain Greek characters
_GREEK_RE = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')

# Perseus editorial citation words embedded as <t> tokens
_CITATION_WORDS = {
    'hom', 'il', 'od', 'ff', 'cf', 'ibid',
    'pp', 'p', 'vol', 'fr', 'frag',
}

# Numeric-only pattern (bare section numbers, line refs like "639-40")
_NUMERIC_RE = re.compile(r'^[\d\-–—.,:;]+$')


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def _is_greek(token: str) -> bool:
    """True if the token contains at least one Greek character."""
    return bool(_GREEK_RE.search(token))


def _is_speaker_label(token: str) -> bool:
    """
    True if the token looks like a speaker abbreviation.

    Heuristic: all uppercase, 1–6 characters, no diacritics.
    Works for ΣΩ, ΙΩΝ, ΚΡ, ΑΠΟΛ, ΕΤ, ΦΑΙ, etc.
    """
    if not token:
        return False
    if len(token) > 6:
        return False
    # NFD: strip combining diacritics, then check all-uppercase
    nfd = unicodedata.normalize('NFD', token)
    base = ''.join(c for c in nfd if not unicodedata.combining(c))
    return base == base.upper() and base.isalpha()


def _is_citation(token: str) -> bool:
    """True if the token is a Perseus editorial citation fragment."""
    return token.lower() in _CITATION_WORDS


def _is_numeric(token: str) -> bool:
    """True if the token is a bare number or numeric reference."""
    return bool(_NUMERIC_RE.match(token))


def filter_tokens(tokens: list[dict],
                  min_chars: int = MIN_CHARS) -> list[str]:
    """
    Filter a raw token list from the scraper JSON.

    Args:
        tokens:    list of {"w": surface_form, "o": offset} dicts
        min_chars: minimum character length to retain

    Returns:
        list of clean surface-form strings (not yet lemmatised)
    """
    clean = []
    for tok in tokens:
        w = tok.get("w", "").strip()
        if not w:
            continue
        if _is_numeric(w):
            continue
        if _is_citation(w):
            continue
        if not _is_greek(w):
            continue
        if _is_speaker_label(w):
            continue
        if len(w) < min_chars:
            continue
        clean.append(w)
    return clean


# ---------------------------------------------------------------------------
# Corpus-level filter
# ---------------------------------------------------------------------------

def filter_corpus(corpus: dict,
                  min_chars: int = MIN_CHARS) -> dict:
    """
    Apply filter_tokens to every document in a corpus dict.

    Args:
        corpus: dict mapping 'author_work' → {'tokens': [...], ...}
                as produced by pipeline.scrape_corpus()

    Returns:
        dict mapping 'author_work' → {'tokens_clean': [...], 'author': ...,
                                       'work': ..., 'token_count_raw': int,
                                       'token_count_clean': int}
    """
    result = {}
    for key, doc in corpus.items():
        raw    = doc.get("tokens", [])
        clean  = filter_tokens(raw, min_chars=min_chars)
        result[key] = {
            "author":            doc.get("author"),
            "work":              doc.get("work"),
            "tokens_clean":      clean,
            "token_count_raw":   len(raw),
            "token_count_clean": len(clean),
        }
        dropped = len(raw) - len(clean)
        pct     = dropped / len(raw) * 100 if raw else 0
        print(f"  {key:<40} {len(raw):>6} → {len(clean):>6} tokens  "
              f"({dropped} dropped, {pct:.1f}%)")
    return result


# ---------------------------------------------------------------------------
# Load from disk (for use after scrape_corpus saves JSONs)
# ---------------------------------------------------------------------------

def load_corpus_from_disk(corpus_dir: str = "greek_corpus") -> dict:
    """
    Load all JSON files saved by scrape_corpus() into a corpus dict.

    Returns:
        dict mapping 'author_work' → raw doc dict (same format as
        scrape_corpus output), ready to pass to filter_corpus()
    """
    corpus = {}
    for path in sorted(Path(corpus_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        key = f"{doc['author']}_{doc['work']}".replace(" ", "_")
        corpus[key] = doc
        print(f"  Loaded {key}  ({len(doc.get('tokens', []))} tokens)")
    return corpus


# ---------------------------------------------------------------------------
# Diagnostic: inspect filtered output for a single work
# ---------------------------------------------------------------------------

def show_filter_sample(tokens_clean: list[str],
                       n: int = 50) -> None:
    """Print the first n clean tokens for a sanity check."""
    print(f"\nFirst {n} clean tokens:")
    print(" ".join(tokens_clean[:n]))
    print(f"\nTotal: {len(tokens_clean)} tokens")