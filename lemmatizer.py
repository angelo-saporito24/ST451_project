"""
Lemmatizer — Stage 2 of the LDA Preprocessing Pipeline
========================================================
Takes clean surface-form token lists (output of token_filter.py) and
produces three lemma representations for each document:

  MAP     — single deterministic lemma per token (top_parse.lemma)
  Weighted — fractional lemma counts proportional to posterior P(lemma|word)
  Sampled  — S stochastic lemma assignments drawn from the posterior

Each representation feeds a different DTM construction strategy in Stage 3.

The morphological posterior comes from MorphAnalyzer (morph_analyzer.py),
which combines Perseids Morpheus (rule-based) and optionally Stanza (neural).

Usage:
    from lemmatizer import Lemmatizer

    lm = Lemmatizer(cache_path="lemma_cache.json")
    result = lm.lemmatize_corpus(clean_corpus, n_samples=50)
    # result[key] has 'map_lemmas', 'weighted_counts', 'sampled_lemmas'
"""

import json
import time
import math
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional

from morph_analyzer import MorphAnalyzer


# ---------------------------------------------------------------------------
# Lemma cache  — avoid re-calling the Perseids API for seen surface forms
# ---------------------------------------------------------------------------

class LemmaCache:
    """
    Persistent JSON cache mapping surface form → lemma distribution.

    Stored as:
        { "surface_form": {"lemma1": prob1, "lemma2": prob2, ...}, ... }

    This is the single most important optimisation: the corpus has ~277k
    tokens but far fewer unique surface forms. The Ion alone has 3652 tokens
    but only ~1200 unique forms. Caching eliminates redundant API calls.
    """

    def __init__(self, path: str = "lemma_cache.json"):
        self.path = Path(path)
        self._cache: dict[str, dict[str, float]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._cache = json.load(f)
            print(f"[LemmaCache] Loaded {len(self._cache)} cached forms "
                  f"from {self.path}")

    def get(self, surface: str) -> Optional[dict[str, float]]:
        return self._cache.get(surface)

    def set(self, surface: str, dist: dict[str, float]) -> None:
        self._cache[surface] = dist

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Lemma distribution extractor
# ---------------------------------------------------------------------------

def _lemma_distribution(word_analysis) -> dict[str, float]:
    """
    Collapse a WordAnalysis parse posterior into a lemma distribution.

    Sums probabilities across all parses sharing the same lemma, returning
    a normalised dict {lemma: probability}.  This is lemma entropy — the
    philosophically significant ambiguity (different words, not different
    inflections of the same word).
    """
    if not word_analysis.parses:
        return {word_analysis.surface: 1.0}

    lemma_probs: dict[str, float] = defaultdict(float)
    for p in word_analysis.parses:
        lemma_probs[p.lemma] += p.prob

    # Normalise (should already sum to ~1.0 but floating point drift)
    total = sum(lemma_probs.values()) or 1.0
    return {l: prob / total for l, prob in lemma_probs.items()}


# ---------------------------------------------------------------------------
# Main lemmatizer
# ---------------------------------------------------------------------------

class Lemmatizer:

    def __init__(self,
                 cache_path:    str   = "lemma_cache.json",
                 morph_delay:   float = 0.6,
                 stanza_weight: float = 0.6,
                 save_every:    int   = 500):
        """
        Args:
            cache_path:    path to persistent lemma cache JSON
            morph_delay:   seconds between Perseids API calls
            stanza_weight: α in the Bayesian mixture (ignored if no Stanza)
            save_every:    save cache to disk every N new analyses
        """
        self.cache      = LemmaCache(cache_path)
        self.morph      = MorphAnalyzer(delay=morph_delay,
                                        stanza_weight=stanza_weight)
        self.save_every = save_every

    # ------------------------------------------------------------------ #
    # Single token                                                         #
    # ------------------------------------------------------------------ #

    def lemma_dist(self, surface: str) -> dict[str, float]:
        """
        Return the lemma probability distribution for a surface form,
        using the cache where possible.
        """
        cached = self.cache.get(surface)
        if cached is not None:
            return cached

        analysis = self.morph._analyze_word(surface)
        dist     = _lemma_distribution(analysis)
        self.cache.set(surface, dist)
        return dist

    def map_lemma(self, surface: str) -> str:
        """Return the single most probable lemma (MAP estimate)."""
        dist = self.lemma_dist(surface)
        return max(dist, key=dist.get)

    def sample_lemma(self, surface: str, rng: random.Random) -> str:
        """Draw one lemma from the posterior distribution."""
        dist  = self.lemma_dist(surface)
        lemmas, probs = zip(*dist.items())
        return rng.choices(lemmas, weights=probs, k=1)[0]

    # ------------------------------------------------------------------ #
    # Document-level                                                       #
    # ------------------------------------------------------------------ #

    def lemmatize_map(self, tokens: list[str]) -> list[str]:
        """
        MAP lemmatization: one deterministic lemma per token.
        Returns a list of lemma strings.
        """
        return [self.map_lemma(t) for t in tokens]

    def lemmatize_weighted(self, tokens: list[str]) -> dict[str, float]:
        """
        Weighted lemmatization: fractional lemma counts.

        Returns a dict {lemma: fractional_count} — the expected count of
        each lemma under the posterior, i.e. sum over tokens of P(lemma|token).
        """
        counts: dict[str, float] = defaultdict(float)
        for t in tokens:
            for lemma, prob in self.lemma_dist(t).items():
                counts[lemma] += prob
        return dict(counts)

    def lemmatize_sampled(self, tokens: list[str],
                          n_samples: int = 50,
                          seed: int = 42) -> list[list[str]]:
        """
        Sampled lemmatization: S stochastic lemma assignments.

        Returns a list of S lemma lists, each drawn independently from
        the posterior.  Each list is one Monte Carlo DTM sample.
        """
        rng     = random.Random(seed)
        samples = []
        for _ in range(n_samples):
            sample = [self.sample_lemma(t, rng) for t in tokens]
            samples.append(sample)
        return samples

    # ------------------------------------------------------------------ #
    # Corpus-level                                                         #
    # ------------------------------------------------------------------ #

    def lemmatize_corpus(self,
                         clean_corpus: dict,
                         n_samples:    int = 50,
                         seed:         int = 42) -> dict:
        """
        Lemmatize every document in the clean corpus under all three methods.

        Args:
            clean_corpus: output of token_filter.filter_corpus()
            n_samples:    number of Monte Carlo samples for sampled method
            seed:         random seed for reproducibility

        Returns:
            dict mapping key → {
                'author':          str,
                'work':            str,
                'map_lemmas':      list[str],
                'weighted_counts': dict[str, float],
                'sampled_lemmas':  list[list[str]],   # shape: (n_samples, n_tokens)
                'n_tokens':        int,
                'n_unique_raw':    int,
                'n_unique_lemmas': int,
                'mean_lemma_entropy': float,
                'n_ambiguous':     int,
            }
        """
        results = {}
        new_analyses = 0

        for key, doc in clean_corpus.items():
            tokens   = doc["tokens_clean"]
            n_unique = len(set(tokens))
            print(f"\n── {key}  ({len(tokens)} tokens, {n_unique} unique) ──")

            # Pre-populate cache for all unique forms in this document
            # (avoids redundant lookups in the three passes below)
            unique_tokens = list(set(tokens))
            for i, t in enumerate(unique_tokens):
                if self.cache.get(t) is None:
                    self.lemma_dist(t)   # populates cache as side effect
                    new_analyses += 1
                    if new_analyses % self.save_every == 0:
                        self.cache.save()
                        print(f"  [cache] saved at {new_analyses} new analyses")
                if (i + 1) % 100 == 0:
                    print(f"  analysed {i+1}/{n_unique} unique forms…",
                          end="\r")
            print(f"  analysed {n_unique}/{n_unique} unique forms    ")

            # Three lemmatization passes (all use cache — no extra API calls)
            map_lemmas      = self.lemmatize_map(tokens)
            weighted_counts = self.lemmatize_weighted(tokens)
            sampled_lemmas  = self.lemmatize_sampled(tokens, n_samples, seed)

            # Entropy diagnostics
            entropies  = []
            n_ambig    = 0
            for t in set(tokens):
                dist = self.lemma_dist(t)
                h    = -sum(p * math.log2(p) for p in dist.values() if p > 0)
                entropies.append(h)
                if h > 0.8:
                    n_ambig += 1
            mean_h = sum(entropies) / len(entropies) if entropies else 0.0

            results[key] = {
                "author":             doc["author"],
                "work":               doc["work"],
                "map_lemmas":         map_lemmas,
                "weighted_counts":    weighted_counts,
                "sampled_lemmas":     sampled_lemmas,
                "n_tokens":           len(tokens),
                "n_unique_raw":       n_unique,
                "n_unique_lemmas":    len(set(map_lemmas)),
                "mean_lemma_entropy": round(mean_h, 3),
                "n_ambiguous":        n_ambig,
            }

            print(f"  MAP lemmas: {len(set(map_lemmas))} unique  |  "
                  f"mean lemma entropy: {mean_h:.3f} bits  |  "
                  f"ambiguous forms: {n_ambig}/{n_unique}")

        # Final cache save
        self.cache.save()
        print(f"\n[LemmaCache] {len(self.cache)} total cached forms saved.")
        return results