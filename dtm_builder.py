"""
DTM Builder — Stage 3 of the LDA Preprocessing Pipeline
=========================================================
Builds document-term matrices from lemmatized corpus output under
three methods:

  Baseline  — MAP lemmas, integer counts, standard sparse DTM
  Weighted  — fractional lemma counts from posterior expectations
  Sampled   — S integer DTMs, one per Monte Carlo sample

Also handles:
  - Stopword removal
  - Frequency-based vocabulary pruning (min_df, max_df)
  - Document chunking (splits large works into smaller units for LDA)

Usage:
    from dtm_builder import build_dtms, chunk_corpus

    dtms = build_dtms(lemmatized_corpus, clean_corpus,
                      n_samples=50, min_df=3, max_df=0.9)
"""

import json
import math
import random
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix, lil_matrix

from greek_stopwords import remove_stopwords, remove_stopwords_weighted
from lemmatizer    import Lemmatizer


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocabulary(lemmatized_corpus: dict,
                     min_df:  int   = 3,
                     max_df:  float = 0.9) -> list[str]:
    """
    Build a shared vocabulary from MAP lemmas across the corpus.

    Args:
        lemmatized_corpus: output of Lemmatizer.lemmatize_corpus()
        min_df:  minimum document frequency (drop lemmas in fewer docs)
        max_df:  maximum document frequency as fraction (drop stopwords
                 not caught by the explicit list)

    Returns:
        sorted list of vocabulary lemmas
    """
    n_docs = len(lemmatized_corpus)
    doc_freq: dict[str, int] = defaultdict(int)
    for doc in lemmatized_corpus.values():
        lemmas_clean = remove_stopwords(doc["map_lemmas"])
        for lemma in set(lemmas_clean):
            doc_freq[lemma] += 1

    vocab = [
        lemma for lemma, df in doc_freq.items()
        if df >= min_df and df / n_docs <= max_df
    ]
    return sorted(vocab)


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

def chunk_document(lemmas: list[str],
                   chunk_size: int = 300,
                   min_chunk:  int = 100) -> list[list[str]]:
    """
    Split a long lemma list into fixed-size chunks for LDA.

    Chunking gives the model sub-document resolution — important for
    detecting within-work conceptual shifts (e.g. early vs. late Republic).

    Args:
        lemmas:     list of lemmas for one document
        chunk_size: target tokens per chunk
        min_chunk:  minimum tokens; shorter final chunks are merged back

    Returns:
        list of lemma lists, each approximately chunk_size tokens
    """
    if len(lemmas) <= chunk_size:
        return [lemmas]

    chunks = [lemmas[i:i + chunk_size]
              for i in range(0, len(lemmas), chunk_size)]

    # Merge final chunk into previous if too short
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk:
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]

    return chunks


def chunk_corpus(lemmatized_corpus: dict,
                 chunk_size: int = 300,
                 min_chunk:  int = 100) -> dict:
    """
    Chunk every document in the corpus and return a flat doc dict.

    Each chunk becomes an independent document for LDA, keyed as
    'author_work_chunk_N'.  Metadata (author, work) is preserved.

    Returns:
        dict mapping chunk_key → {
            'author', 'work', 'map_lemmas',
            'weighted_counts', 'chunk_idx', 'n_chunks'
        }
    """
    chunked = {}
    for key, doc in lemmatized_corpus.items():
        map_chunks = chunk_document(doc["map_lemmas"],
                                    chunk_size, min_chunk)
        n = len(map_chunks)
        for i, chunk_lemmas in enumerate(map_chunks):
            w_counts  = Counter(chunk_lemmas)
            chunk_key = f"{key}_chunk_{i:03d}"
            chunked[chunk_key] = {
                "author":          doc["author"],
                "work":            doc["work"],
                "chunk_idx":       i,
                "n_chunks":        n,
                "map_lemmas":      chunk_lemmas,
                "weighted_counts": dict(w_counts),
            }
        print(f"  {key:<45} → {n} chunks  "
              f"(~{len(doc['map_lemmas']) // n} lemmas/chunk)")
    return chunked


# ---------------------------------------------------------------------------
# DTM construction helpers
# ---------------------------------------------------------------------------

def _lemmas_to_bow(lemmas: list[str],
                   vocab:  dict[str, int]) -> dict[int, int]:
    """Convert a lemma list to a bag-of-words dict {vocab_idx: count}."""
    bow: dict[str, int] = defaultdict(int)
    for l in remove_stopwords(lemmas):
        if l in vocab:
            bow[vocab[l]] += 1
    return bow


def _weighted_to_bow(weighted: dict[str, float],
                     vocab:    dict[str, int]) -> dict[int, float]:
    """Convert weighted counts to a fractional bow {vocab_idx: frac_count}."""
    clean = remove_stopwords_weighted(weighted)
    return {vocab[l]: c for l, c in clean.items() if l in vocab}


# ---------------------------------------------------------------------------
# Main DTM builder
# ---------------------------------------------------------------------------

def build_dtms(lemmatized_corpus: dict,
               clean_corpus:      dict,
               n_samples:         int   = 50,
               chunk_size:        int   = 300,
               min_chunk:         int   = 100,
               min_df:            int   = 3,
               max_df:            float = 0.9,
               seed:              int   = 42,
               cache_path:        str   = "lemma_cache.json") -> dict:
    """
    Build all three DTMs from the lemmatized corpus.

    Args:
        lemmatized_corpus: output of Lemmatizer.lemmatize_corpus()
        clean_corpus:      output of filter_corpus() — for sampled method
        n_samples:         Monte Carlo samples for sampled DTM
        chunk_size:        tokens per chunk (0 = no chunking)
        min_chunk:         minimum chunk size
        min_df / max_df:   vocabulary frequency thresholds
        seed:              random seed
        cache_path:        lemma cache path for sampled regeneration

    Returns:
        dict with keys:
            'vocab'          : list[str]
            'vocab_index'    : dict[str, int]
            'doc_keys'       : list[str]
            'doc_meta'       : list[dict]
            'dtm_baseline'   : np.ndarray  shape (n_docs, vocab_size) int
            'dtm_weighted'   : np.ndarray  shape (n_docs, vocab_size) float
            'dtm_samples'    : np.ndarray  shape (n_samples, n_docs, vocab_size) int
    """
    # ── 1. Chunk corpus ──────────────────────────────────────────────────
    print("── Chunking corpus ──")
    if chunk_size > 0:
        chunked = chunk_corpus(lemmatized_corpus, chunk_size, min_chunk)
    else:
        chunked = {k: {**v, "chunk_idx": 0, "n_chunks": 1}
                   for k, v in lemmatized_corpus.items()}
    n_docs = len(chunked)
    print(f"\n  Total chunks (documents for LDA): {n_docs}")

    # ── 2. Build vocabulary ──────────────────────────────────────────────
    print("\n── Building vocabulary ──")
    vocab_list  = build_vocabulary(lemmatized_corpus, min_df, max_df)
    vocab_index = {l: i for i, l in enumerate(vocab_list)}
    V = len(vocab_list)
    print(f"  Vocabulary size: {V} lemmas  "
          f"(min_df={min_df}, max_df={max_df})")

    # ── 3. Document ordering ─────────────────────────────────────────────
    doc_keys = sorted(chunked.keys())
    doc_meta = [{"author":    chunked[k]["author"],
                 "work":      chunked[k]["work"],
                 "chunk_idx": chunked[k]["chunk_idx"],
                 "n_chunks":  chunked[k]["n_chunks"]}
                for k in doc_keys]

    # ── 4. Baseline DTM (MAP, integer counts) ────────────────────────────
    print("\n── Building baseline DTM (MAP) ──")
    dtm_baseline = np.zeros((n_docs, V), dtype=np.int32)
    for d, k in enumerate(doc_keys):
        for idx, cnt in _lemmas_to_bow(chunked[k]["map_lemmas"],
                                       vocab_index).items():
            dtm_baseline[d, idx] = cnt
    print(f"  Shape: {dtm_baseline.shape}  "
          f"Sparsity: {(dtm_baseline == 0).mean():.1%}")

    # ── 5. Weighted DTM (fractional counts) ──────────────────────────────
    print("\n── Building weighted DTM ──")
    dtm_weighted = np.zeros((n_docs, V), dtype=np.float32)
    for d, k in enumerate(doc_keys):
        for idx, cnt in _weighted_to_bow(chunked[k]["weighted_counts"],
                                         vocab_index).items():
            dtm_weighted[d, idx] = cnt
    print(f"  Shape: {dtm_weighted.shape}  "
          f"Sparsity: {(dtm_weighted == 0).mean():.1%}")

    # ── 6. Sampled DTMs (Monte Carlo) ────────────────────────────────────
    print(f"\n── Building sampled DTMs ({n_samples} samples) ──")
    lm  = Lemmatizer(cache_path=cache_path)
    rng = random.Random(seed)
    dtm_samples = np.zeros((n_samples, n_docs, V), dtype=np.int32)

    for s in range(n_samples):
        for d, k in enumerate(doc_keys):
            corpus_key = k.rsplit("_chunk_", 1)[0] if "_chunk_" in k else k
            raw_tokens = clean_corpus[corpus_key]["tokens_clean"]
            meta       = doc_meta[d]
            cs         = chunk_size if chunk_size > 0 else len(raw_tokens)
            start      = meta["chunk_idx"] * cs
            tok_chunk  = raw_tokens[start: start + cs]
            sampled    = [lm.sample_lemma(t, rng) for t in tok_chunk]
            for idx, cnt in _lemmas_to_bow(sampled, vocab_index).items():
                dtm_samples[s, d, idx] = cnt
        if (s + 1) % 10 == 0:
            print(f"  sample {s+1}/{n_samples}", end="\r")
    print(f"  Shape: {dtm_samples.shape}  "
          f"Sparsity: {(dtm_samples == 0).mean():.1%}  ✓")

    return {
        "vocab":        vocab_list,
        "vocab_index":  vocab_index,
        "doc_keys":     doc_keys,
        "doc_meta":     doc_meta,
        "dtm_baseline": dtm_baseline,
        "dtm_weighted": dtm_weighted,
        "dtm_samples":  dtm_samples,
    }