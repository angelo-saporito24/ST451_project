"""
Greek Text Analysis Pipeline
=============================
Unified interface combining:
  1. PerseusScraper    — fetch Greek + English from Scaife/Perseus
  2. Passage.find_greek — cross-lingual alignment (LaBSE or Jaccard)
  3. MorphAnalyzer     — Bayesian morphological breakdown
  4. Commentary        — Claude-powered commentary grounded in parse posteriors
  5. scrape_corpus()   — bulk corpus scraping for BML/LDA pipeline

Usage:
    p = GreekPipeline()
    result = p.run("Phaedrus", "243", "a god or something divine")
    print(result)

    # External translation (Jowett etc.)
    result = p.run("Phaedrus", "243", "Love is a god",
                   own_translation=True, commentary=True)

    # Bulk corpus scrape
    corpus = p.scrape_corpus(corpus_spec, output_dir="greek_corpus")

Dependencies:
    pip install requests beautifulsoup4 lxml
    pip install sentence-transformers   # optional: LaBSE alignment
    pip install stanza                  # optional: neural morphology
"""

import os
import json
import time
import textwrap
import requests
from dataclasses import dataclass, field
from typing import Optional

from perseus_scraper import PerseusScraper, Passage, AlignmentResult
from morph_analyzer  import MorphAnalyzer, PhraseAnalysis


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    work:       str
    reference:  str
    query:      str
    passage:       Optional[Passage]        = None
    greek_match:   Optional[str]            = None
    english_match: Optional[str]            = None
    morph:         Optional[PhraseAnalysis] = None
    commentary:    Optional[str]            = None
    elapsed:       dict = field(default_factory=dict)

    def __str__(self) -> str:
        bar  = "═" * 62
        thin = "─" * 62
        lines = [
            f"\n{bar}",
            f"  GREEK TEXT PIPELINE",
            f"  {self.work.title()} · {self.reference}  |  query: \"{self.query}\"",
            f"{bar}",
        ]
        if self.passage:
            lines += [
                "", "── PASSAGE ─────────────────────────────────────────────────",
                "", "  GREEK:",
                textwrap.indent(self.passage.greek_text   or "[none]", "    "),
                "", "  ENGLISH:",
                textwrap.indent(self.passage.english_text or "[none]", "    "),
            ]
        lines += ["", thin, "  ALIGNMENT", thin]
        if self.greek_match:
            lines += [
                f"  English match : {self.english_match}",
                f"  Greek match   : {self.greek_match}",
            ]
        else:
            lines.append("  [alignment failed]")
        lines += ["", thin, "  MORPHOLOGICAL ANALYSIS", thin]
        if self.morph:
            lines.append(str(self.morph))
        else:
            lines.append("  [no morph analysis]")
        if self.commentary:
            lines += ["", thin, "  COMMENTARY (Claude)", thin, ""]
            lines.append(textwrap.indent(self.commentary, "  "))
        if self.elapsed:
            lines += ["", thin]
            for stage, t in self.elapsed.items():
                lines.append(f"  ⏱  {stage:<22} {t:.1f}s")
        lines.append(f"\n{bar}\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commentary generator
# ---------------------------------------------------------------------------

def _build_commentary_prompt(greek_phrase, english_phrase, morph,
                              work, reference):
    morph_lines = []
    for w in morph.words:
        top = w.top_parse
        if not top:
            continue
        line = f"  • {w.surface}: lemma={top.lemma}, {top.pos}"
        if top.features:
            line += f", {top.feature_str()}"
        if w.lemma_entropy > 0.3:
            alts = [p.lemma for p in w.parses[1:] if p.lemma != top.lemma][:2]
            if alts:
                line += f"  [ambiguous — also: {', '.join(alts)}]"
        if w.gloss:
            line += f"\n    ◆ {w.gloss}"
        morph_lines.append(line)

    system = (
        "You are a scholarly commentator on ancient Greek philosophy. "
        "You write with precision, philosophical depth, and economy. "
        "You never pad. You always ground claims in the grammar. "
        "You are familiar with Plato, Aristotle, the Presocratics, and "
        "their modern interpreters (Heidegger, Gadamer, Nussbaum, etc.)."
    )
    user = f"""
You are commenting on this Greek phrase from {work} ({reference}):

  Greek   : {greek_phrase}
  English : {english_phrase}

The morphological analysis (with Bayesian parse posteriors) is:

{chr(10).join(morph_lines)}

Write a focused commentary of 200–300 words covering:
1. Literal grammatical structure — what each key word is doing syntactically.
2. Philosophical weight — what the choice of these words (not paraphrases) carries.
3. Any ambiguity flagged above that is philosophically significant.
4. One connection to another text or argument where relevant.

Do not use bullet points. Write in connected prose. Do not pad or summarise.
""".strip()
    return system, user


def generate_commentary(greek_phrase, english_phrase, morph,
                        work, reference, api_key=None):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[Commentary] ANTHROPIC_API_KEY not set — skipping.")
        return None
    system, user = _build_commentary_prompt(
        greek_phrase, english_phrase, morph, work, reference
    )
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-6",
                "max_tokens": 600,
                "system":     system,
                "messages":   [{"role": "user", "content": user}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    except Exception as e:
        print(f"[Commentary] API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class GreekPipeline:

    def __init__(self, scraper_delay=1.5, morph_delay=0.7,
                 stanza_weight=0.6, anthropic_key=None):
        self.scraper = PerseusScraper(delay=scraper_delay)
        self.morph   = MorphAnalyzer(delay=morph_delay,
                                     stanza_weight=stanza_weight)
        self.api_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")

    # ------------------------------------------------------------------ #
    # Single-passage analysis                                              #
    # ------------------------------------------------------------------ #

    def run(self, work, reference, query, end_ref=None, author="plato",
            commentary=True, show_full_passage=True,
            own_translation=False) -> PipelineResult:
        """
        Run all four stages for a single passage.

        Args:
            work:             e.g. "Phaedrus", "Medea"
            reference:        Stephanus ref e.g. "243", "243b", or line "1"
            query:            English phrase to align (any translation)
            end_ref:          if given, fetches range reference–end_ref
            author:           default "plato"
            commentary:       whether to call Claude for commentary
            show_full_passage: include full Greek+English in output
            own_translation:  True if query is from a non-Perseus translation
        """
        result = PipelineResult(work=work, reference=reference, query=query)

        # Stage 1: Scrape
        t0 = time.time()
        print(f"[1/4] Fetching {work} {reference}" +
              (f"–{end_ref}" if end_ref else "") + "…")
        try:
            passage = (self.scraper.get_passage_range(work, reference, end_ref,
                                                       author=author)
                       if end_ref
                       else self.scraper.get_passage(work, reference,
                                                     author=author))
            result.passage = passage
        except Exception as e:
            print(f"  [Scrape error] {e}")
            return result
        result.elapsed["1. scrape"] = time.time() - t0

        # Stage 2: Align
        t0 = time.time()
        print(f"[2/4] Aligning \"{query}\"" +
              (" [external translation]" if own_translation else "") + "…")
        aligned = (self._align_external(passage, query)
                   if own_translation
                   else passage.find_greek(query))
        if aligned:
            result.greek_match   = aligned.greek_text
            result.english_match = aligned.english_text
            print(f"  → Greek: {aligned.greek_text}")
        else:
            print("  [Alignment failed]")
            return result
        result.elapsed["2. align"] = time.time() - t0

        # Stage 3: Morphology
        t0 = time.time()
        print(f"[3/4] Analysing morphology…")
        result.morph = self.morph.analyze(result.greek_match)
        result.elapsed["3. morphology"] = time.time() - t0

        # Stage 4: Commentary
        if commentary and self.api_key:
            t0 = time.time()
            print("[4/4] Generating commentary…")
            result.commentary = generate_commentary(
                result.greek_match, result.english_match,
                result.morph, work, reference, self.api_key,
            )
            result.elapsed["4. commentary"] = time.time() - t0
        else:
            if commentary:
                print("[4/4] Skipping commentary (no API key).")
            result.elapsed["4. commentary"] = 0.0

        if not show_full_passage:
            result.passage = None
        return result

    def analyze_phrase(self, greek, english="", work="unknown",
                       reference="", commentary=True) -> PipelineResult:
        """Analyse a Greek phrase directly — no scraping needed."""
        result = PipelineResult(work=work, reference=reference,
                                query=greek, greek_match=greek,
                                english_match=english)
        t0 = time.time()
        print(f"[1/2] Analysing: {greek}")
        result.morph = self.morph.analyze(greek)
        result.elapsed["1. morphology"] = time.time() - t0

        if commentary and self.api_key and english:
            t0 = time.time()
            print("[2/2] Generating commentary…")
            result.commentary = generate_commentary(
                greek, english, result.morph, work, reference, self.api_key,
            )
            result.elapsed["2. commentary"] = time.time() - t0
        return result

    # ------------------------------------------------------------------ #
    # External translation alignment                                       #
    # ------------------------------------------------------------------ #

    def _align_external(self, passage: Passage, query: str):
        try:
            from sentence_transformers import SentenceTransformer
            print("  [LaBSE] using semantic alignment…")
            return passage.find_greek(query)
        except ImportError:
            pass
        if not self.api_key:
            print("  [Warning] No LaBSE or API key — falling back to Jaccard.")
            return passage.find_greek(query)
        print("  [Claude] using API for cross-translation alignment…")
        return self._align_via_claude(passage, query)

    def _align_via_claude(self, passage: Passage, query: str):
        prompt = (
            f"You are a Greek scholar. A student is reading in their own "
            f"translation and wants to find the corresponding Greek.\n\n"
            f"Greek passage:\n{passage.greek_text or ''}\n\n"
            f"The student's phrase: \"{query}\"\n\n"
            f"Reply with ONLY the corresponding Greek words, nothing else."
        )
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":    "claude-sonnet-4-6",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=20,
            )
            resp.raise_for_status()
            greek_match = resp.json()["content"][0]["text"].strip()
            return AlignmentResult(
                query=query,
                english_tokens=query.split(),
                greek_tokens=greek_match.split(),
                english_text=query,
                greek_text=greek_match,
                confidence=0.85,
                method="Claude cross-translation alignment",
            )
        except Exception as e:
            print(f"  [Claude alignment error] {e}")
            return passage.find_greek(query)

    # ------------------------------------------------------------------ #
    # Corpus scraping (for BML / LDA pipeline)                            #
    # ------------------------------------------------------------------ #

    def scrape_corpus(self, corpus_spec: list[dict],
                      output_dir: str = "corpus") -> dict:
        """
        Scrape a full corpus and save each document to disk as JSON.

        Args:
            corpus_spec: list of dicts with keys 'author', 'work',
                         'sections' (list of refs, or 'auto' to fetch all)
            output_dir:  directory to save JSON files

        Returns:
            dict mapping 'author_work' → {'greek', 'tokens', ...}

        Example:
            corpus_spec = [
                {'author': 'plato',    'work': 'phaedrus', 'sections': 'auto'},
                {'author': 'euripides','work': 'bacchae',  'sections': 'auto'},
            ]
            corpus = pipeline.scrape_corpus(corpus_spec)
        """
        os.makedirs(output_dir, exist_ok=True)
        corpus = {}

        for spec in corpus_spec:
            author   = spec['author']
            work     = spec['work']
            sections = spec.get('sections', 'auto')
            key      = f"{author}_{work}".replace(' ', '_')

            print(f"\n── Scraping {author.title()} · {work.title()} ──")

            if sections == 'auto':
                sections = self._get_valid_sections(author, work)
                if not sections:
                    print(f"  [!] Could not retrieve section list — skipping.")
                    continue

            all_greek, all_tokens = [], []
            for sec in sections:
                try:
                    p = self.scraper.get_passage(work, sec, author=author)
                    if p.greek_text:
                        all_greek.append(p.greek_text)
                    if p.greek_tokens:
                        all_tokens.extend(p.greek_tokens)
                    print(f"  ✓ {sec}", end="\r")
                    time.sleep(self.scraper.delay)
                except Exception as e:
                    print(f"  [!] {sec}: {e}")

            doc = {
                'author':  author,
                'work':    work,
                'greek':   '\n\n'.join(all_greek),
                'tokens':  all_tokens,
            }
            corpus[key] = doc

            fpath = os.path.join(output_dir, f"{key}.json")
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            print(f"\n  Saved → {fpath}  ({len(all_tokens)} tokens)")

        return corpus

    def _get_valid_sections(self, author: str, work: str) -> list[str]:
        from perseus_scraper import get_all_sections
        return get_all_sections(author, work)