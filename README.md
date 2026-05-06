# ST451 — Bayesian NLP Pipeline for Ancient Greek Philosophical Topic Modelling

**Angelo Saporito | MSc Data Science, LSE | May 2026**

This repository contains the full analysis pipeline for the ST451 Bayesian Machine Learning project: probabilistic preprocessing and topic uncertainty quantification in Ancient Greek philosophical texts.

---

## Project Overview

The project constructs a Bayesian NLP pipeline for topic modelling twelve works of Plato and Aristotle, treating morphological analysis as a probabilistic inference problem. Rather than relying on a single deterministic document-term matrix, three representations are compared: a baseline MAP DTM, a posterior-weighted DTM, and an ensemble of 50 Monte Carlo sampled DTMs. Latent Dirichlet Allocation is fitted under each representation to measure how preprocessing uncertainty propagates into downstream topic structure. The Septuagint (Genesis, Proverbs, Wisdom of Solomon, Sirach) serves as a held-out projection corpus.

---

## Repository Structure

```
ST451_Project/
├── 71322_ST451_Project.ipynb   # unified analysis notebook (submission)
├── README.md                   # this file
├── requirements.txt            # Python dependencies
├── .gitignore
│
├── greek_corpus/               # scraped raw JSON files, one per work (16 works)
├── lemma_cache.json            # 44,879 cached Morpheus API analyses
├── lemmatized_corpus.json      # MAP lemmas + weighted counts for all 16 works
├── dtm_baseline.npy            # baseline MAP document-term matrix (1133 x 1336)
├── dtm_weighted.npy            # posterior-weighted DTM (1133 x 1336)
├── dtm_meta.json               # vocabulary, document keys, and metadata
│
├── pipeline.py                 # top-level scraping pipeline
├── token_filter.py             # corpus loading and token filtering
├── lemmatizer.py               # Morpheus API lemmatisation with caching
├── dtm_builder.py              # DTM construction (baseline, weighted, sampled)
├── lda_fitter.py               # LDA fitting across all three DTMs
├── morph_analyzer.py           # morphological analysis utilities
├── perseus_scraper.py          # Perseus/Scaife CTS API scraper
└── greek_stopwords.py          # Ancient Greek stopword list
```

---

## Run Order

The notebook is self-contained and must be executed **top to bottom in a single session**. It is divided into four sections:

1. **Corpus scraping and lemmatisation** — loads from cached artefacts if present
2. **DTM construction and LDA fitting** — builds all three DTMs and fits LDA at K=15
3. **Figures 1–10** — produces all main analysis figures
4. **Extensions C, A, E, F** — register entropy, Septuagint analysis, coherence, covariate regression

### Running with cached artefacts (recommended)

With `FORCE_RESCRAPE = False` (default), the notebook skips all scraping and API calls if the following artefacts are present in the working directory:

- `greek_corpus/` — all JSON corpus files
- `lemma_cache.json` — Morpheus API cache
- `lemmatized_corpus.json` — lemmatised corpus

This allows the full pipeline to run without any API access.

### Running from scratch

Set `FORCE_RESCRAPE = True` in Cell 1.1. Note that the full scrape and lemmatisation takes several hours due to Perseids API rate limiting (44,879 unique surface forms must be analysed).

### Note on `dtm_samples.npy`

The sampled DTM (`dtm_samples.npy`, shape 50×1133×1336, ~300MB) is excluded from this repository due to file size. It is regenerated automatically when Cell 2.3 runs. `dtm_baseline.npy` and `dtm_weighted.npy` are included and will be loaded directly if present.

---

## Figure Output

All figures are saved to a `figures/` subdirectory (created automatically on first run). The directory contains 30 files: PDF and PNG versions of all 10 main figures and 5 extension figures.

---

## Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key packages: `numpy`, `scikit-learn`, `matplotlib`, `scipy`, `networkx`, `beautifulsoup4`, `requests`, `lxml`.

---

## Data Sources

- **Plato and Aristotle:** Perseus Digital Library via the Scaife CTS API (https://scaife.perseus.org/)
- **Septuagint:** greekdoc.github.io (interlinear Greek/English table)
- **Morphological analysis:** Perseids Morpheus API (https://morph.perseids.org/)

---

## AI Acknowledgement

In accordance with ST451 Position 2 guidelines, Claude (Anthropic, claude-sonnet-4-5, accessed April–May 2026) was used to assist with notebook harmonisation and light prose compression. All analytical decisions and substantive content are the author's own. Full acknowledgement is included in the submitted PDF and notebook.
