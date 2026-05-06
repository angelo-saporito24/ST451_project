
"""
Bayesian Greek Morphological Analyzer
...
"""
import re
import math
"""
Bayesian Greek Morphological Analyzer
======================================
Combines two sources to build a posterior distribution P(parse | word):

  Source A — Stanza (primary, neural):
    A BiLSTM+CRF model trained on the PROIEL Ancient Greek treebank.
    Returns a single best-parse per word with a confidence score.
    Works entirely offline with Unicode Greek input.
    Install: pip install stanza
             python -c "import stanza; stanza.download('grc')"

  Source B — Perseids Morpheus API (secondary, rule-based):
    The original Perseus Morpheus engine. Returns multiple possible
    parsings per word with weights. Requires Beta Code input — we
    convert Unicode → Beta Code internally.
    No extra install needed (uses requests).

The two sources are combined via a simple mixture:
    P(parse | word) ∝ α · P_stanza(parse) + (1-α) · P_perseids(parse)

Shannon entropy over this posterior flags genuinely ambiguous forms.

Usage:
    analyzer = MorphAnalyzer()
    result   = analyzer.analyze("θεὸς ἤ τι θεῖον ὁ Ἔρως")
    print(result)

Dependencies:
    pip install requests stanza      # stanza optional but recommended
    python -c "import stanza; stanza.download('grc')"
"""

import re
import math
import time
import unicodedata
import requests
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Optional Stanza import
# ---------------------------------------------------------------------------

try:
    import stanza
    _STANZA_AVAILABLE = True
except ImportError:
    _STANZA_AVAILABLE = False
    print("[MorphAnalyzer] stanza not installed — using Perseids API only.")
    print("  To enable neural parsing: pip install stanza")
    print("  Then: python -c \"import stanza; stanza.download('grc')\"")


# ---------------------------------------------------------------------------
# Beta Code conversion  (Unicode polytonic → Perseus Beta Code)
# ---------------------------------------------------------------------------

_BASE = {
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e',
    'ζ': 'z', 'η': 'h', 'θ': 'q', 'ι': 'i', 'κ': 'k',
    'λ': 'l', 'μ': 'm', 'ν': 'n', 'ξ': 'c', 'ο': 'o',
    'π': 'p', 'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't',
    'υ': 'u', 'φ': 'f', 'χ': 'x', 'ψ': 'y', 'ω': 'w',
}

# Combining diacritics → Beta Code symbols (order matters: breathing, accent, subscript)
_DIAC = {
    '\u0313': ')',   # smooth breathing (comma above)
    '\u0314': '(',   # rough breathing (reversed comma above)
    '\u0301': '/',   # acute accent
    '\u0300': '\\',  # grave accent
    '\u0342': '=',   # circumflex (perispomeni)
    '\u0308': '+',   # diaeresis
    '\u0345': '|',   # iota subscript (ypogegrammeni)
}

# Beta Code output order: breathing → accent → subscript
_DIAC_ORDER = {')': 0, '(': 0, '/': 1, '\\': 1, '=': 1, '+': 2, '|': 3}


def unicode_to_betacode(word: str) -> str:
    """
    Convert a polytonic Unicode Greek word to Perseus Beta Code.

    Example: θεὸς → qeo\\s,  ἄνθρωπος → a)/nqrwpos
    """
    result = []
    # NFD decomposes precomposed characters into base + combining marks
    nfd = unicodedata.normalize('NFD', word.lower())
    i = 0
    while i < len(nfd):
        ch = nfd[i]
        if ch in _BASE:
            letter = _BASE[ch]
            # Collect all following combining characters
            j = i + 1
            diacritics = []
            while j < len(nfd) and unicodedata.combining(nfd[j]) > 0:
                bc = _DIAC.get(nfd[j])
                if bc:
                    diacritics.append(bc)
                j += 1
            # Sort in Beta Code order: breathing, accent, subscript
            diacritics.sort(key=lambda d: _DIAC_ORDER.get(d, 9))
            result.append(letter + ''.join(diacritics))
            i = j
        else:
            i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Philosophical glossary
# ---------------------------------------------------------------------------

PHILOSOPHICAL_GLOSSARY = {
    'θεός':      'god; the divine ground of reality in Plato',
    'θεῖος':     'divine, god-given; adjectival form of θεός',
    'ἔρως':      'Eros; desire, love — both cosmic force and personal passion',
    'τέχνη':     'craft, art, skill; systematic transferable knowledge',
    'μοῖρα':     'portion, fate, allotment — θείᾳ μοίρᾳ: by divine allotment',
    'λόγος':     'word, reason, rational account, argument',
    'ψυχή':      'soul; immortal, self-moving principle of life',
    'εἶδος':     'form, kind; Platonic Form in the metaphysical sense',
    'ἀνάμνησις': 'recollection; Platonic doctrine of knowledge as recovery',
    'μανία':     'madness; in Phaedrus, divine madness superior to sanity',
    'κάλλος':    'beauty; the most visible of the Forms',
    'ἀρετή':     'virtue, excellence',
    'δαίμων':    'divine intermediary between gods and humans',
    'ποίησις':   'making, creation; poetic production',
    'κτῆμα':     'possession, property — τοῦτο τὸ κτῆμα: this (divine) possession',
    'ἔνθεος':    'enthused, god-within; the state of divine seizure',
    'χρησμός':   'oracle, divine utterance',
    'μοῦσα':     'Muse; divine source of poetic inspiration',
    'ἁλμυρός':   'briny, salty — metaphor for the bitter effect of bad speech',
    'ἀκοή':      'hearing, the act of listening',
    'καθαρμός':  'purification, ritual cleansing',
    'παλινῳδία': 'palinode; a recantation poem — Stesichorus\'s and Socrates\'s',
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Parse:
    lemma:    str
    pos:      str
    features: dict
    prob:     float = 0.0
    source:   str   = ''     # 'stanza', 'perseids', or 'combined'

    def feature_str(self) -> str:
        order = ('tense', 'mood', 'voice', 'person', 'number', 'case', 'gender', 'degree')
        parts = [f"{k}: {self.features[k]}" for k in order if k in self.features]
        return ', '.join(parts) if parts else '—'

    def __str__(self):
        return (f"  [{self.source}]  lemma={self.lemma}, "
                f"pos={self.pos}, {self.feature_str()},  P={self.prob:.1%}")


@dataclass
class WordAnalysis:
    surface:      str
    parses:       list[Parse]
    entropy:      float
    is_ambiguous: bool
    gloss:         Optional[str]
    betacode:      str   = ''
    lemma_entropy: float = 0.0   # entropy over lemmas only (lexical ambiguity)

    @property
    def top_parse(self) -> Optional[Parse]:
        return self.parses[0] if self.parses else None

    def ambiguity_label(self) -> str:
        # Use lemma_entropy so inflection variants don't inflate the warning
        if   self.lemma_entropy < 0.3: return '✓ unambiguous'
        elif self.lemma_entropy < 1.0: return '~ mildly ambiguous'
        else:                          return '⚠ lexically ambiguous'

    def n_distinct_lemmas(self) -> int:
        return len({p.lemma for p in self.parses})

    def __str__(self):
        lines = [f"\n{'─'*52}"]
        amb   = self.ambiguity_label()
        lines.append(
            f"  {self.surface}  [{amb}]"
            f"  H(lemma)={self.lemma_entropy:.2f} bits"
            f"  H(form)={self.entropy:.2f} bits"
        )
        if not self.parses:
            lines.append('    [no analysis returned]')
            return '\n'.join(lines)
        top = self.top_parse
        lines.append(
            f"  Best parse ({top.prob:.0%}): {top.lemma} | {top.pos} | {top.feature_str()}"
        )
        if self.n_distinct_lemmas() > 1:
            # Show alternatives only when there is genuine lexical ambiguity
            lines.append(f"  Lexical alternatives ({self.n_distinct_lemmas()} possible lemmas):")
            seen = set()
            for p in self.parses[1:]:
                if p.lemma not in seen:
                    seen.add(p.lemma)
                    # Sum probability across all forms of this lemma
                    lemma_prob = sum(q.prob for q in self.parses if q.lemma == p.lemma)
                    lines.append(f"    {lemma_prob:5.1%}  {p.lemma} | {p.pos}")
        elif len(self.parses) > 1:
            # Same lemma, multiple forms — just note the inflection uncertainty
            cases = list({p.features.get('case', '?') for p in self.parses})
            lines.append(f"  Inflection uncertainty: case may be {' / '.join(cases)}")
        if self.gloss:
            lines.append(f"  ◆  {self.gloss}")
        return '\n'.join(lines)


@dataclass
class PhraseAnalysis:
    phrase:       str
    words:        list[WordAnalysis]
    mean_entropy: float = 0.0
    n_ambiguous:  int   = 0

    def __str__(self):
        bar = '═' * 57
        lines = [
            f'\n{bar}',
            f'  Phrase: {self.phrase}',
            f'  Words: {len(self.words)}  |  '
            f'Ambiguous: {self.n_ambiguous}  |  '
            f'Mean entropy: {self.mean_entropy:.2f} bits',
            bar,
        ]
        for w in self.words:
            lines.append(str(w))
        lines.append(f'\n{bar}\n')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Stanza wrapper
# ---------------------------------------------------------------------------

class StanzaParser:
    """
    Wraps a Stanza Ancient Greek NLP pipeline.
    Lazy-loads the model on first use (downloads ~500MB once).
    """

    def __init__(self):
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            print('[Stanza] Loading Ancient Greek model (first call only)…')
            self._nlp = stanza.Pipeline(
                lang='grc',
                processors='tokenize,pos,lemma,depparse',
                tokenize_pretokenized=True,   # we pass pre-split tokens
                verbose=False,
            )

    def parse_word(self, word: str) -> Optional[Parse]:
        """Return a single Parse for one word, confidence = 1.0 (Stanza is deterministic)."""
        if not _STANZA_AVAILABLE:
            return None
        self._load()
        doc = self._nlp([[word]])
        if not doc.sentences or not doc.sentences[0].words:
            return None
        w = doc.sentences[0].words[0]
        features = _parse_ufeats(w.feats or '')
        return Parse(
            lemma=w.lemma or word,
            pos=w.upos or '',
            features=features,
            prob=1.0,   # placeholder; will be re-weighted in mixer
            source='stanza',
        )


def _parse_ufeats(feats_str: str) -> dict:
    """
    Parse a Universal Dependencies feature string like
    'Case=Nom|Gender=Masc|Number=Sing' into a dict.
    """
    if not feats_str or feats_str == '_':
        return {}
    result = {}
    key_map = {
        'Case':   'case',   'Gender': 'gender', 'Number': 'number',
        'Tense':  'tense',  'Mood':   'mood',   'Voice':  'voice',
        'Person': 'person', 'Degree': 'degree',
    }
    for part in feats_str.split('|'):
        if '=' in part:
            k, v = part.split('=', 1)
            if k in key_map:
                result[key_map[k]] = v.lower()
    return result


# ---------------------------------------------------------------------------
# Perseids API wrapper
# ---------------------------------------------------------------------------

class PerseidsParser:
    """
    Calls morph.perseids.org after converting Unicode → Beta Code.
    Returns multiple Parse candidates from the rule-based Morpheus engine.
    """

    URL = ('https://morph.perseids.org/analysis/word'
           '?lang=grc&engine=morpheusgrc&word={word}')

    def __init__(self, delay: float = 0.6):
        self.delay   = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GreekMorphAnalyzer/2.0 (academic)',
            'Accept':     'application/json',
        })

    def parse_word(self, word: str) -> list[Parse]:
        """Return un-normalized Parse list for a word."""
        bc  = unicode_to_betacode(word)
        url = self.URL.format(word=requests.utils.quote(bc))
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code not in (200, 201):
                return []
            return self._parse_json(resp.json())
        except (requests.RequestException, ValueError):
            return []

    @staticmethod
    def _parse_json(data: dict) -> list[Parse]:
        parses = []
        try:
            annotation = data['RDF']['Annotation']
            body = annotation.get('Body', [])
            if isinstance(body, dict):
                body = [body]
        except (KeyError, TypeError):
            return []

        for entry_block in body:
            rest  = entry_block.get('rest', {})
            entry = rest.get('entry', {})

            # Lemma
            hdwd = entry.get('dict', {}).get('hdwd', {})
            lemma = hdwd.get('$', '') if isinstance(hdwd, dict) else str(hdwd)

            # POS
            pofs = entry.get('dict', {}).get('pofs', {})
            pos  = pofs.get('$', '') if isinstance(pofs, dict) else str(pofs)

            # Weight (default 1.0 per entry if not specified)
            weight = float(entry_block.get('wt', 1.0))

            # Inflection forms
            infl_list = entry.get('infl', [])
            if isinstance(infl_list, dict):
                infl_list = [infl_list]

            for infl in infl_list:
                features = _parse_perseids_infl(infl)
                # Override pos from infl if available
                infl_pos = infl.get('pofs', {})
                infl_pos_str = (infl_pos.get('$', pos)
                                if isinstance(infl_pos, dict) else str(infl_pos))
                parses.append(Parse(
                    lemma=lemma,
                    pos=infl_pos_str or pos,
                    features=features,
                    prob=weight,
                    source='perseids',
                ))
        return parses


def _parse_perseids_infl(infl: dict) -> dict:
    key_map = {
        'tense': 'tense', 'mood': 'mood', 'voice': 'voice',
        'pers':  'person', 'num':  'number', 'case': 'case',
        'gend':  'gender', 'comp': 'degree',
    }
    out = {}
    for api_key, label in key_map.items():
        val = infl.get(api_key, {})
        if isinstance(val, dict):
            val = val.get('$', '')
        if val:
            out[label] = val.lower()
    return out


# ---------------------------------------------------------------------------
# Bayesian mixer
# ---------------------------------------------------------------------------

def mix_parses(stanza_parse: Optional[Parse],
               perseids_parses: list[Parse],
               alpha: float = 0.6) -> list[Parse]:
    """
    Combine Stanza (weight α) and Perseids (weight 1-α) into a joint posterior.

    If both sources agree on lemma+pos, their probabilities add.
    If only one source is available, we use it with a uniform prior on the other.
    """
    if not stanza_parse and not perseids_parses:
        return []

    # Normalise Perseids weights
    if perseids_parses:
        total = sum(p.prob for p in perseids_parses) or 1.0
        for p in perseids_parses:
            p.prob = p.prob / total

    # Build combined distribution keyed by (lemma, pos, frozenset(features))
    combined: dict[tuple, Parse] = {}

    if stanza_parse:
        key = (stanza_parse.lemma, stanza_parse.pos,
               frozenset(stanza_parse.features.items()))
        stanza_parse.prob = alpha
        combined[key] = stanza_parse

    if perseids_parses:
        scale = (1 - alpha) if stanza_parse else 1.0
        for p in perseids_parses:
            key = (p.lemma, p.pos, frozenset(p.features.items()))
            if key in combined:
                combined[key].prob += scale * p.prob
                combined[key].source = 'combined'
            else:
                p.prob = scale * p.prob
                combined[key] = p

    result = sorted(combined.values(), key=lambda p: p.prob, reverse=True)

    # Re-normalise to sum to 1
    total = sum(p.prob for p in result) or 1.0
    for p in result:
        p.prob /= total

    return result


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class MorphAnalyzer:
    """
    Bayesian morphological analyzer combining Stanza (neural) and
    Perseids Morpheus (rule-based).
    """

    AMBIGUITY_THRESHOLD = 0.8   # bits

    def __init__(self, delay: float = 0.6, stanza_weight: float = 0.6):
        """
        Args:
            delay:          seconds between Perseids API calls
            stanza_weight:  α in the mixture; weight given to Stanza's parse
        """
        self.alpha    = stanza_weight
        self._stanza  = StanzaParser() if _STANZA_AVAILABLE else None
        self._perseids = PerseidsParser(delay=delay)

    def analyze(self, phrase: str) -> PhraseAnalysis:
        tokens = self._tokenize(phrase)
        words  = []
        for i, tok in enumerate(tokens):
            words.append(self._analyze_word(tok))
            if i < len(tokens) - 1:
                time.sleep(self._perseids.delay)

        ents    = [w.entropy for w in words]
        mean_h  = sum(ents) / len(ents) if ents else 0.0
        n_amb   = sum(1 for w in words if w.is_ambiguous)
        return PhraseAnalysis(phrase=phrase, words=words,
                              mean_entropy=mean_h, n_ambiguous=n_amb)

    def _analyze_word(self, word: str) -> WordAnalysis:
        stanza_p   = self._stanza.parse_word(word) if self._stanza else None
        perseids_p = self._perseids.parse_word(word)
        parses     = mix_parses(stanza_p, perseids_p, self.alpha)
        entropy    = _entropy(parses)
        gloss      = _gloss(parses)
        lem_h      = _lemma_entropy(parses)
        bc         = unicode_to_betacode(word)
        return WordAnalysis(
            surface=word, parses=parses, entropy=entropy,
            is_ambiguous=(lem_h > self.AMBIGUITY_THRESHOLD),
            gloss=gloss, betacode=bc, lemma_entropy=lem_h,
        )

    @staticmethod
    def _tokenize(phrase: str) -> list[str]:
        greek_re = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]+')
        tokens   = []
        for raw in phrase.split():
            cleaned = re.sub(r'[.,·;:—\-–!?\[\]()"\']+', '', raw)
            if greek_re.search(cleaned):
                tokens.append(cleaned)
        return tokens


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _entropy(parses: list[Parse]) -> float:
    h = 0.0
    for p in parses:
        if p.prob > 0:
            h -= p.prob * math.log2(p.prob)
    return h


def _gloss(parses: list[Parse]) -> Optional[str]:
    """
    Search all parses (not just top-1) for a glossary match.
    This ensures we catch cases like θεῖον where the top parse
    is a false positive (θέω) but a lower parse has θεῖος.
    Priority: exact match on any lemma > prefix match on best lemma.
    """
    if not parses:
        return None
    # First pass: exact match across all lemmas, weighted by probability
    for p in parses:
        lemma = p.lemma.strip()
        if lemma in PHILOSOPHICAL_GLOSSARY:
            return PHILOSOPHICAL_GLOSSARY[lemma]
    # Second pass: prefix match on top parse only
    lemma = parses[0].lemma.strip()
    for key, gloss in PHILOSOPHICAL_GLOSSARY.items():
        if lemma and key and lemma[:4] == key[:4]:
            return gloss
    return None


def _lemma_entropy(parses: list[Parse]) -> float:
    """
    Entropy over the *lemma* distribution, collapsing inflection variants.
    Distinguishes true lexical ambiguity (different words) from inflection
    uncertainty (same word, different case/number/etc.).
    """
    lemma_probs: dict[str, float] = {}
    for p in parses:
        lemma_probs[p.lemma] = lemma_probs.get(p.lemma, 0.0) + p.prob
    h = 0.0
    for prob in lemma_probs.values():
        if prob > 0:
            h -= prob * math.log2(prob)
    return h
