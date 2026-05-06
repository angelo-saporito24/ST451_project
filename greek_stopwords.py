"""
Ancient Greek Stopword List
============================
Function words that should be removed before LDA topic modelling.
These are grammatical words that carry no topical information:
particles, conjunctions, prepositions, pronouns, common auxiliaries,
Socratic dialogue mechanics, and elided/clipped forms unresolved by
the lemmatizer.

Crucially, philosophically loaded words are NOT stopwords even if
frequent: λόγος, ψυχή, εἶδος, ἀρετή, τέχνη, νοῦς, θεός etc.
These are the signal, not the noise.

The list targets lemma forms (output of MorphAnalyzer), not surface forms.
Exception: the elided forms block targets unresolved surface forms that
survive lemmatization unchanged.
"""

GREEK_STOPWORDS = {
    # ── Particles and conjunctions ──────────────────────────────────────
    "καί", "δέ", "τε", "γάρ", "ἀλλά", "οὖν", "μέν", "ἤ", "ὅτι",
    "εἰ", "ὡς", "ἵνα", "ὅτε", "ἐπεί", "ὅταν", "ἐάν", "ἄν",
    "μή", "οὐ", "οὐκ", "οὐχ", "οὔτε", "μήτε", "μηδέ", "οὐδέ",
    "οὐδείς", "μηδείς", "μήν", "ἆρα", "εἴτε", "ὅμως", "καίτοι",
    "ἤτοι", "ἤδη", "ἔτι", "αὖ", "αὖθις", "πάλιν", "ποτέ", "πώ",
    "πώποτε", "νῦν", "τότε", "εὐθύς", "ἅμα", "ἄρτι",

    # ── Prepositions ────────────────────────────────────────────────────
    "ἐν", "ἐκ", "εἰς", "ἐξ", "πρός", "παρά", "ὑπό",
    "ἐπί", "ἀπό", "διά", "μετά", "κατά", "ἀντί", "ἀνά", "σύν",
    "ὑπέρ", "πρό", "ἄνευ", "ἕνεκα", "πλήν",

    # ── Pronouns ────────────────────────────────────────────────────────
    "αὐτός", "ἐγώ", "σύ", "ἡμεῖς", "ὑμεῖς", "τις", "τί",
    "ὅς", "ὅστις", "ἐκεῖνος", "οὗτος", "ὅδε", "τοιοῦτος",
    "τοσοῦτος", "ἄλλος", "ἄλλοτε", "ἀλλήλων", "ἑαυτοῦ",
    "ὁ", "ἡ", "τό",   # article as lemma

    # ── Common verbs (copula, auxiliaries, high-frequency) ───────────────
    "εἰμί", "γίγνομαι", "ἔχω", "λέγω", "εἶπον", "φημί",
    "ὁράω", "ἐρωτάω", "ἀποκρίνομαι", "δοκεῖ", "δεῖ", "χρή",
    "δύναμαι", "δίδωμι", "οἴομαι",
    "ὁμολογέω", "διαλέγω", "κελεύω",
    "εἶμι",   # "to go" — high-frequency, no topical content

    # ── Socratic dialogue mechanics (particles / discourse markers) ───────
    "οὐκοῦν", "ἆρʼ", "οἴμη", "τοίνυν", "δήπου",

    # ── Conversational hedges and dialogue fillers ───────────────────────
    # High-frequency Platonic softeners / filler terms with no
    # propositional content. Distinct from philosophically loaded words.
    "πού",          # "somehow / I suppose" — epistemic hedge
    "παντάπασι",    # "altogether / entirely" — intensifier
    "γέλοιος",      # "ridiculous" — evaluative filler in dialogue
    "μάλη",         # variant filler form
    "νυνδί",        # "just now" — temporal discourse marker
    "πη",           # "somehow / in some way" — epistemic hedge

    # ── Adverbs (non-topical) ────────────────────────────────────────────
    "οὕτως", "ὥσπερ", "ὅλως", "ὅλος", "εὖ", "μάλα",
    "μᾶλλον", "μάλιστα", "ἧττον", "ἥκιστα", "πάνυ", "σφόδρα",
    "ἀεί", "ποτέ", "ὁμοίως", "ὡσαύτως", "ἴσως",

    # ── Numerals and quantifiers ─────────────────────────────────────────
    "εἷς", "δύο", "τρεῖς", "πᾶς", "ὅλος", "πολύς", "ὀλίγος",
    "ἕκαστος", "ἄμφω", "ἑκάτερος",

    # ── Elided / clipped forms (unresolved by lemmatizer) ────────────────
    # Surface forms of particles/conjunctions that survive lemmatization
    # unchanged. Placed here rather than token_filter.py to avoid
    # stripping elided forms that the lemmatizer does resolve correctly.
    "ἀλλʼ", "δʼ", "καθʼ", "διʼ", "γʼ", "τʼ", "τοῦτʼ", "ἐπʼ",
    "ὑπʼ", "ἀπʼ", "κατʼ", "μετʼ", "παρʼ", "πρόσʼ",
}


def remove_stopwords(lemmas: list[str],
                     extra:  set[str] = None) -> list[str]:
    """
    Remove stopwords from a list of lemmas.

    Args:
        lemmas: list of lemma strings (MAP or sampled)
        extra:  optional additional stopwords to remove

    Returns:
        filtered list with stopwords removed
    """
    stops = GREEK_STOPWORDS | (extra or set())
    return [l for l in lemmas if l not in stops]


def remove_stopwords_weighted(weighted_counts: dict[str, float],
                               extra: set[str] = None) -> dict[str, float]:
    """
    Remove stopwords from a weighted count dict.
    """
    stops = GREEK_STOPWORDS | (extra or set())
    return {l: c for l, c in weighted_counts.items() if l not in stops}