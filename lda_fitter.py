"""
LDA Fitter — Stage 4 of the LDA Preprocessing Pipeline
========================================================
Fits Latent Dirichlet Allocation under three methods:

  Baseline  — single LDA on MAP DTM
  Weighted  — single LDA on fractional-count DTM
  Sampled   — S LDA fits on S sampled DTMs, then aggregated

For each method returns:
  - topic-word matrix  phi:    (K, V)  — P(word | topic)
  - doc-topic matrix   theta:  (D, K)  — P(topic | document)

The sampled method additionally returns:
  - phi_samples:   (S, K, V) — one phi per sample
  - theta_samples: (S, D, K) — one theta per sample
  - phi_std:       (K, V)    — std of phi across samples
  - theta_std:     (D, K)    — std of theta across samples

The std matrices are the BML contribution: they quantify how much
morphological ambiguity destabilises downstream topic structure.

Usage:
    from lda_fitter import fit_all, select_k

    # Find optimal K
    scores = select_k(dtms, k_range=range(5, 25))

    # Fit all three methods at chosen K
    results = fit_all(dtms, K=15)
"""

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# K selection via perplexity
# ---------------------------------------------------------------------------

def select_k(dtms:    dict,
             k_range: range = range(5, 26, 5),
             n_iter:  int   = 100,
             n_jobs:  int   = 1,
             seed:    int   = 42) -> dict:
    """
    Fit baseline LDA across a range of K values and return perplexity.

    Lower perplexity = better fit, but watch for overfitting.
    Use alongside topic coherence (inspected manually) to pick K.

    Args:
        dtms:    output of build_dtms()
        k_range: iterable of K values to try
        n_jobs:  parallel jobs for LDA (-1 = all cores)
        seed:    random state

    Returns:
        dict mapping K → perplexity score
    """
    X      = dtms["dtm_baseline"]
    scores = {}
    print("── K selection (baseline DTM, perplexity) ──")
    for K in k_range:
        lda = LatentDirichletAllocation(
            n_components      = K,
            max_iter          = n_iter,
            learning_method   = "online",
            batch_size        = 128,
            learning_offset   = 10.0,
            learning_decay    = 0.7,
            random_state      = seed,
            n_jobs            = n_jobs,
        )
        lda.fit(X)
        perp = lda.perplexity(X)
        scores[K] = perp
        print(f"  K={K:2d}  perplexity={perp:.1f}")
    return scores


# ---------------------------------------------------------------------------
# Single LDA fit
# ---------------------------------------------------------------------------

def _fit_lda(X:       np.ndarray,
             K:       int,
             n_iter:  int = 100,
             seed:    int = 42,
             n_jobs:  int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit one LDA model and return normalised phi and theta.

    Returns:
        phi:   (K, V) topic-word distributions
        theta: (D, K) document-topic distributions
    """
    lda = LatentDirichletAllocation(
        n_components    = K,
        max_iter        = n_iter,
        learning_method = "online",
        batch_size      = 128,
        learning_offset = 10.0,
        learning_decay  = 0.7,
        random_state    = seed,
        n_jobs          = n_jobs,
    )
    lda.fit(X)

    # phi: normalise rows of components_ to sum to 1
    phi = lda.components_ / lda.components_.sum(axis=1, keepdims=True)

    # theta: transform X through fitted model
    theta = lda.transform(X)   # already row-normalised by sklearn

    return phi, theta, lda


# ---------------------------------------------------------------------------
# Topic alignment across samples
# ---------------------------------------------------------------------------

def _align_topics(phi_ref:     np.ndarray,
                  phi_sample:  np.ndarray) -> np.ndarray:
    """
    Align topics in phi_sample to phi_ref by greedy cosine similarity.

    Necessary for the sampled method: topics may be permuted across
    LDA fits, so we need to match them before computing variance.

    Returns:
        phi_aligned: (K, V) with rows permuted to match phi_ref
    """
    from scipy.optimize import linear_sum_assignment

    # Cosine similarity matrix (K x K)
    ref_norm  = phi_ref    / (np.linalg.norm(phi_ref,    axis=1, keepdims=True) + 1e-10)
    samp_norm = phi_sample / (np.linalg.norm(phi_sample, axis=1, keepdims=True) + 1e-10)
    sim       = ref_norm @ samp_norm.T   # (K, K)

    # Hungarian algorithm: maximise similarity = minimise -similarity
    row_ind, col_ind = linear_sum_assignment(-sim)
    return phi_sample[col_ind]


# ---------------------------------------------------------------------------
# Fit all three methods
# ---------------------------------------------------------------------------

def fit_all(dtms:   dict,
            K:      int   = 15,
            n_iter: int   = 100,
            seed:   int   = 42,
            n_jobs: int   = 1) -> dict:
    """
    Fit LDA under all three methods and return results.

    Args:
        dtms:   output of build_dtms()
        K:      number of topics
        n_iter: LDA iterations
        seed:   random state
        n_jobs: parallel jobs

    Returns:
        dict with keys:
            'K', 'vocab', 'doc_keys', 'doc_meta'

            # Baseline
            'phi_baseline':   (K, V)
            'theta_baseline': (D, K)

            # Weighted
            'phi_weighted':   (K, V)
            'theta_weighted': (D, K)

            # Sampled — mean and uncertainty
            'phi_sampled':    (K, V)   mean phi across samples
            'theta_sampled':  (D, K)   mean theta across samples
            'phi_std':        (K, V)   std of phi across samples
            'theta_std':      (D, K)   std of theta across samples
    """
    vocab    = dtms["vocab"]
    doc_keys = dtms["doc_keys"]
    doc_meta = dtms["doc_meta"]
    S        = dtms["dtm_samples"].shape[0]

    # ── Baseline ─────────────────────────────────────────────────────────
    print(f"── Fitting baseline LDA  (K={K}) ──")
    phi_b, theta_b, _ = _fit_lda(dtms["dtm_baseline"],
                                  K, n_iter, seed, n_jobs)
    print(f"  phi:   {phi_b.shape}   theta: {theta_b.shape}")

    # ── Weighted ─────────────────────────────────────────────────────────
    print(f"\n── Fitting weighted LDA  (K={K}) ──")
    phi_w, theta_w, _ = _fit_lda(dtms["dtm_weighted"].astype(np.float64),
                                  K, n_iter, seed, n_jobs)
    print(f"  phi:   {phi_w.shape}   theta: {theta_w.shape}")

    # ── Sampled ───────────────────────────────────────────────────────────
    print(f"\n── Fitting sampled LDA  (K={K}, S={S}) ──")
    phi_s_all   = np.zeros((S, K, len(vocab)))
    theta_s_all = np.zeros((S, len(doc_keys), K))

    for s in range(S):
        X_s            = dtms["dtm_samples"][s]
        phi_s, theta_s, _ = _fit_lda(X_s, K, n_iter, seed + s, n_jobs)
        # Align to baseline phi to handle topic permutation
        phi_s_aligned       = _align_topics(phi_b, phi_s)
        phi_s_all[s]        = phi_s_aligned
        # Align theta columns to match
        K_ref = phi_b.shape[0]
        ref_norm  = phi_b   / (np.linalg.norm(phi_b,   axis=1, keepdims=True) + 1e-10)
        samp_norm = phi_s   / (np.linalg.norm(phi_s,   axis=1, keepdims=True) + 1e-10)
        from scipy.optimize import linear_sum_assignment
        _, col_ind          = linear_sum_assignment(-(ref_norm @ samp_norm.T))
        theta_s_all[s]      = theta_s[:, col_ind]
        if (s + 1) % 10 == 0:
            print(f"  sample {s+1}/{S}", end="\r")
    print(f"  Done.  phi_s_all: {phi_s_all.shape}")

    phi_sampled   = phi_s_all.mean(axis=0)
    theta_sampled = theta_s_all.mean(axis=0)
    phi_std       = phi_s_all.std(axis=0)
    theta_std     = theta_s_all.std(axis=0)

    print(f"\n── Summary ──────────────────────────────────────")
    print(f"  Baseline  phi std (mean): {phi_b.std():.4f}")
    print(f"  Weighted  phi std (mean): {phi_w.std():.4f}")
    print(f"  Sampled   phi std (mean across samples): "
          f"{phi_std.mean():.4f}")

    return {
        "K":        K,
        "vocab":    vocab,
        "doc_keys": doc_keys,
        "doc_meta": doc_meta,

        "phi_baseline":   phi_b,
        "theta_baseline": theta_b,

        "phi_weighted":   phi_w,
        "theta_weighted": theta_w,

        "phi_sampled":    phi_sampled,
        "theta_sampled":  theta_sampled,
        "phi_std":        phi_std,
        "theta_std":      theta_std,
    }


# ---------------------------------------------------------------------------
# Topic inspection helpers
# ---------------------------------------------------------------------------

def top_words(phi: np.ndarray, vocab: list[str],
              topic_idx: int, n: int = 15) -> list[str]:
    """Return top n words for a given topic."""
    idxs = phi[topic_idx].argsort()[::-1][:n]
    return [vocab[i] for i in idxs]


def print_topics(phi: np.ndarray, vocab: list[str],
                 n_words: int = 12, label: str = "") -> None:
    """Print all topics with their top words."""
    K = phi.shape[0]
    print(f"\n── Topics ({label}, K={K}) ──────────────────────")
    for k in range(K):
        words = top_words(phi, vocab, k, n_words)
        print(f"  Topic {k:2d}: {', '.join(words)}")


def topic_stability(phi_std: np.ndarray,
                    vocab:   list[str],
                    K:       int,
                    n:       int = 10) -> None:
    """
    For each topic print the words with highest variance across samples.
    High-variance words are most affected by morphological ambiguity.
    """
    print("\n── Topic instability (high-variance words per topic) ──")
    for k in range(K):
        idxs  = phi_std[k].argsort()[::-1][:n]
        words = [f"{vocab[i]}({phi_std[k,i]:.4f})" for i in idxs]
        print(f"  Topic {k:2d}: {', '.join(words)}")