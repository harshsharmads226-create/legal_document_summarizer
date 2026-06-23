from __future__ import annotations
 
import logging
import re
from typing import IO, List, NamedTuple, Union
 
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
logger = logging.getLogger(__name__)
 
 
# -------------------------
# EXCEPTIONS
# -------------------------
 
class SummarizerError(Exception):
    """Base class for all errors raised by this module."""
 
 
class EmptyDocumentError(SummarizerError):
    """Raised when the input has no usable text content."""
 
 
class PDFExtractionError(SummarizerError):
    """Raised when a PDF can't be opened, or has no extractable text layer."""
 
 
# -------------------------
# RESULT TYPE
# -------------------------
 
class SummaryResult(NamedTuple):
    """Result of summarize_text() / summarize_pdf().
 
    Behaves like a plain 3-tuple for backward compatibility, so existing
    code doing `summary, total, used = summarize_text(text)` keeps working
    unchanged. Also supports named access, e.g. `result.summary`.
    """
    summary: str
    total_sentences: int
    sentences_used: int
 
 
# -------------------------
# NLTK SETUP
# -------------------------
 
def _ensure_nltk_data() -> None:
    """Download required NLTK tokenizer data if missing, with a clear
    error instead of a cryptic LookupError deep inside tokenization if
    there's no internet access to fetch it."""
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as exc:
                raise SummarizerError(
                    f"Required NLTK data package '{pkg}' is missing and could not "
                    f"be downloaded automatically (no internet access?). Run "
                    f"`python -m nltk.downloader {pkg}` manually. Original error: {exc}"
                ) from exc
 
 
_ensure_nltk_data()
 
# Legal abbreviations that NLTK's default punkt model misreads as sentence
# endings (e.g. "Rs. 50,000" gets split into "Rs." / "50,000 ..."), cutting
# real sentences in half.
LEGAL_ABBREVIATIONS = {
    "rs", "no", "nos", "mr", "mrs", "ms", "smt", "shri", "dr",
    "sec", "secs", "art", "arts", "vs", "co", "ltd", "ors", "anr",
    "addl", "spl", "j", "jj", "cr", "civ", "crl", "para", "paras",
    "u/s", "w.p", "s.c", "a.i.r", "i.e", "e.g", "etc", "govt", "appx",
}
 
 
def _load_sentence_tokenizer() -> nltk.tokenize.punkt.PunktSentenceTokenizer:
    """Load the pretrained punkt sentence tokenizer and extend its
    abbreviation list with legal-domain abbreviations, without discarding
    the model's pretrained sentence-boundary statistics."""
    for resource in ("tokenizers/punkt/PY3/english.pickle", "tokenizers/punkt/english.pickle"):
        try:
            tok = nltk.data.load(resource)
            tok._params.abbrev_types.update(LEGAL_ABBREVIATIONS)
            return tok
        except LookupError:
            continue
    raise SummarizerError(
        "Could not load NLTK's punkt sentence tokenizer from any known "
        "resource path. Try running `python -m nltk.downloader punkt`."
    )
 
 
_SENT_TOKENIZER = _load_sentence_tokenizer()
 
 
# -------------------------
# LEGAL MARKERS
# -------------------------
 
VERDICT_MARKERS: List[str] = [
    "appeal dismissed", "appeal is dismissed", "appeal stands dismissed",
    "appeal allowed", "appeal is allowed",
    "petition dismissed", "petition allowed",
    "suit dismissed", "suit decreed",
    "granted decree", "decree for rs",
    "decree granted", "decree is confirmed",
    "judgment is confirmed", "trial court judgment is confirmed",
    "final order", "final judgment",
    "disposed of", "set aside", "upheld", "reversed",
    "modified", "affirmed",
    "relief granted", "relief denied",
    "hereby ordered", "hereby directed",
    "is ordered to", "are ordered to",
    "convicted", "acquitted", "sentenced to",
    "the court holds", "the court finds",
    "we hold that", "we find that",
    "for the foregoing reasons", "in view of the above",
    "in the result", "accordingly",
]
 
# Terse, near-standalone operative lines that ARE the verdict but don't
# contain a full VERDICT_MARKERS phrase (e.g. just "Acquitted."). Checked
# against the whole sentence, not as a substring bonus.
TERSE_VERDICT_PATTERNS: List[str] = [
    r"^appeal\s+dismissed\.?$",
    r"^appeal\s+allowed\.?$",
    r"^petition\s+dismissed\.?$",
    r"^petition\s+allowed\.?$",
    r"^suit\s+dismissed\.?$",
    r"^suit\s+decreed\.?$",
    r"^convicted\.?$",
    r"^acquitted\.?$",
    r"^set\s+aside\.?$",
    r"^upheld\.?$",
    r"^affirmed\.?$",
]
 
FACT_MARKERS: List[str] = [
    "brief facts", "facts of the case",
    "it is the case of", "according to the plaintiffs",
    "the plaintiffs stated", "the defendant contended",
    "it is alleged", "the complainant",
]
 
AUTHORITY_WORDS: List[str] = [
    "shall", "is directed to", "are directed to",
    "is ordered", "are ordered",
    "liable to pay", "entitled to",
]
 
LEGAL_STRUCTURE_MARKERS: List[str] = [
    "section", "rule", "article",
    "code of civil procedure", "cpc",
    "trial court", "appellate court",
    "issues framed", "findings",
]
 
# Common phrasing that mentions a dismissal/allowance but refers to an
# earlier or lower court's ruling, not the final verdict of this judgment.
PROCEDURAL_NOISE: List[str] = [
    "trial court dismissed",
    "sessions court dismissed",
    "lower court dismissed",
    "the application was dismissed",
    "was earlier dismissed",
    "had dismissed",
    "had allowed",
]
 
 
# -------------------------
# HEADER CLEANING
# -------------------------
 
def clean_legal_headers(text: str) -> str:
    """Strip court letterhead boilerplate, divider lines, and paragraph
    numbering from raw judgment text before sentence tokenization."""
    patterns = [
        r"IN THE .*?COURT.*?\n",
        r"Coram:.*?\n",
        r"Appeal Suit No\..*?\n",
        r"Cross-Objection No\..*?\n",
        r"M\.P\.No\..*?\n",
        r"C\.M\.P\.No\..*?\n",
        r"For Appellant.*?\n",
        r"For Respondents.*?\n",
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE | re.DOTALL)
 
    # Strip divider lines (3+ dashes/underscores/equals) one at a time,
    # rather than pairing up any two divider lines and deleting everything
    # between them (which previously deleted entire facts/reasoning
    # sections when a divider appeared more than once in the document).
    text = re.sub(r"(?m)^[\s\-_=]{3,}$\n?", "", text)
 
    # Strip leading paragraph numbers like "12. " only when followed by a
    # capital letter or "(" -- this avoids stripping the leading digits of
    # a statutory section number that happens to start a line, e.g.
    # "302. Whoever commits murder...".
    text = re.sub(r"(?m)^\s*\d{1,3}\.\s+(?=[A-Z(])", "", text)
 
    return text.strip()
 
 
# -------------------------
# TEXT PROCESSING
# -------------------------
 
def tokenize_raw_sentences(text: str) -> List[str]:
    """All sentences, unfiltered by length. Used for verdict extraction so
    short operative lines like 'Appeal dismissed.' aren't lost."""
    return [s.strip() for s in _SENT_TOKENIZER.tokenize(text) if s.strip()]
 
 
def split_into_sentences(text: str, min_words: int = 4) -> List[str]:
    """Sentences used for TF-IDF/PageRank. A small minimum word count is
    kept here purely to keep noise out of the similarity matrix --
    verdict extraction does NOT use this filtered list."""
    return [s for s in tokenize_raw_sentences(text) if len(s.split()) >= min_words]
 
 
def build_similarity_matrix(sentences: List[str]) -> np.ndarray:
    """Cosine-similarity matrix over TF-IDF sentence vectors, used as the
    graph for PageRank."""
    if not sentences:
        return np.zeros((0, 0))
    try:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError as exc:
        # Can happen on pathological input where every sentence reduces to
        # nothing after stopword removal. Degrade gracefully instead of
        # crashing -- ranking will then rely on legal_bonus alone.
        logger.warning(
            "TF-IDF vectorization failed (%s); falling back to a zero "
            "similarity matrix.", exc,
        )
        return np.zeros((len(sentences), len(sentences)))
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)
    return sim
 
 
def pagerank(sim_matrix: np.ndarray, d: float = 0.85, max_iter: int = 50, tol: float = 1e-4) -> np.ndarray:
    """Standard power-iteration PageRank over a sentence similarity graph."""
    n = sim_matrix.shape[0]
    if n == 0:
        return np.array([])
    row_sum = sim_matrix.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    M = sim_matrix / row_sum
    scores = np.ones(n) / n
    for _ in range(max_iter):
        prev = scores.copy()
        scores = (1 - d) / n + d * M.T.dot(scores)
        if np.linalg.norm(scores - prev, 1) < tol:
            break
    return scores
 
 
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize PageRank scores to [0, 1].
 
    Raw PageRank scores average ~1/n and shrink as documents get longer
    (e.g. ~0.01 on a 100-sentence document), while legal_bonus terms are
    flat values up to ~0.8-0.9. Without normalizing, PageRank contributes
    almost nothing to ranking on anything but very short documents --
    putting both signals on a comparable scale lets PageRank's centrality
    signal actually influence which sentences get picked.
    """
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
 
 
# -------------------------
# LEGAL HEURISTICS
# -------------------------
 
def legal_bonus(sentence: str, index: int, total: int) -> float:
    """Domain-specific score adjustment layered on top of PageRank, based
    on legal keyword markers, sentence position, and sentence length."""
    text = sentence.lower()
    bonus = 0.0
 
    if any(m in text for m in VERDICT_MARKERS):
        bonus += 0.40
    if any(f in text for f in FACT_MARKERS):
        bonus += 0.25
    if any(a in text for a in AUTHORITY_WORDS):
        bonus += 0.15
    if any(s in text for s in LEGAL_STRUCTURE_MARKERS):
        bonus += 0.10
 
    # Last 20% of document gets a strong boost (verdict zone).
    if index / total > 0.80:
        bonus += 0.20
 
    if 18 <= len(sentence.split()) <= 40:
        bonus += 0.10
 
    if re.search(r"rs\.?\s?\d+", text):
        bonus += 0.15
 
    # Penalize lawyer arguments -- not the court's holding.
    if "learned counsel" in text or "submitted that" in text:
        bonus -= 0.20
 
    # Penalize procedural noise so the verdict fallback doesn't grab it.
    if any(p in text for p in PROCEDURAL_NOISE):
        bonus -= 0.30
 
    return bonus
 
 
# -------------------------
# CLASSIFICATION
# -------------------------
 
def classify_sentence(sentence: str) -> str:
    """Assign a sentence to one of FACTS / ISSUES / PROCEDURAL HISTORY /
    DECISION-VERDICT / OTHER, in that priority order."""
    s = sentence.lower()
    if any(m in s for m in VERDICT_MARKERS):
        return "DECISION / VERDICT"
    if "whether" in s and ("court" in s or "issue" in s):
        return "ISSUES"
    if any(f in s for f in FACT_MARKERS):
        return "FACTS"
    if ("filed" in s or "preferred" in s or "aggrieved" in s) and \
       ("appeal" in s or "petition" in s or "suit" in s):
        return "PROCEDURAL HISTORY"
    return "OTHER"
 
 
# -------------------------
# VERDICT EXTRACTION
# -------------------------
 
def extract_verdict_sentences(raw_sentences: List[str]) -> List[str]:
    """Find verdict/operative sentences, best first.
 
    Runs on the UNFILTERED sentence list so short terse operative lines
    ('Acquitted.') aren't invisible to it, and checks
    TERSE_VERDICT_PATTERNS in addition to the longer VERDICT_MARKERS
    phrases. Prefers sentences from the last 30% of the document, and
    filters out PROCEDURAL_NOISE matches (references to an earlier/lower
    court's ruling, not this judgment's final verdict).
    """
    total = len(raw_sentences)
    if total == 0:
        return []
 
    verdicts = []
    for i, s in enumerate(raw_sentences):
        sl = s.lower().strip()
        is_marker = any(m in sl for m in VERDICT_MARKERS)
        is_terse = any(re.match(p, sl) for p in TERSE_VERDICT_PATTERNS)
        if (is_marker or is_terse) and not any(p in sl for p in PROCEDURAL_NOISE):
            pos_score = 1.0 + (0.5 if i / total > 0.70 else 0.0)
            if is_terse:
                pos_score += 0.25  # an explicit terse operative line is the strongest signal
            verdicts.append((pos_score, i, s))
 
    verdicts.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [s for _, _, s in verdicts]
 
 
# -------------------------
# SENTENCE SELECTION
# -------------------------
 
def pick_top_sentences(
    sentences: List[str], scores: np.ndarray, max_sentences: int
) -> List[tuple]:
    """Rank sentences by normalized PageRank score + legal_bonus, and
    return the top `max_sentences` as (original_index, sentence) pairs."""
    total = len(sentences)
    norm_scores = normalize_scores(scores)
    ranked = [
        (norm_scores[i] + legal_bonus(sentences[i], i, total), i, sentences[i])
        for i in range(total)
    ]
    ranked.sort(reverse=True, key=lambda x: x[0])
    return [(idx, sent) for _, idx, sent in ranked[:max_sentences]]
 
 
# -------------------------
# DYNAMIC max_sentences BASED ON DOCUMENT LENGTH
# -------------------------
 
def compute_max_sentences(total_sentences: int, user_max: int, strict_max: bool = True) -> int:
    """Scale the summary length to document length.
 
    If `strict_max` is True (default), the returned value never exceeds
    `user_max` -- it's treated as a hard ceiling, so an explicit request
    for a short summary is always honored. If False, longer documents are
    allowed to produce a longer summary than requested (e.g. doubling for
    a 200+ sentence document), matching the module's original behavior.
    """
    if total_sentences < 30:
        recommended = min(user_max, 5)
    elif total_sentences < 80:
        recommended = user_max
    elif total_sentences < 200:
        recommended = max(user_max, 12)
    else:
        recommended = max(user_max, 18)
 
    if strict_max:
        return min(recommended, user_max)
    return recommended
 
 
# -------------------------
# FINAL SUMMARIZER API
# -------------------------
 
def summarize_text(text: str, max_sentences: int = 8, strict_max: bool = True) -> SummaryResult:
    """Produce a structured extractive summary of a legal/court judgment.
 
    Parameters
    ----------
    text:
        Raw judgment text.
    max_sentences:
        Target number of sentences to rank and consider for the summary
        (the final summary is further capped per-section -- see `limits`
        below -- so the actual output is usually shorter than this).
    strict_max:
        If True, `max_sentences` is a hard ceiling even on long documents.
        If False, long documents may produce more sentences than asked
        for. See `compute_max_sentences`.
 
    Returns
    -------
    SummaryResult
        Named tuple (summary, total_sentences, sentences_used). Unpacks
        like a plain tuple for backward compatibility.
 
    Raises
    ------
    TypeError
        If `text` is not a string.
    ValueError
        If `max_sentences` is not a positive integer.
    EmptyDocumentError
        If `text` is empty or whitespace-only.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    if not isinstance(max_sentences, int) or max_sentences < 1:
        raise ValueError(f"max_sentences must be a positive integer, got {max_sentences!r}")
    if not text.strip():
        raise EmptyDocumentError("Input text is empty or whitespace-only.")
 
    text = clean_legal_headers(text)
 
    # Raw (unfiltered) sentences feed verdict extraction so short operative
    # lines aren't lost; filtered sentences feed TF-IDF/PageRank.
    raw_sentences = tokenize_raw_sentences(text)
    sentences = split_into_sentences(text)
 
    if not raw_sentences:
        # Text had content but nothing tokenizable as a sentence (e.g.
        # just numbers/punctuation) -- graceful empty result, not an
        # error, since the input itself was valid.
        logger.warning("No sentences could be tokenized after cleaning the input text.")
        return SummaryResult(summary="", total_sentences=0, sentences_used=0)
 
    max_sentences = compute_max_sentences(
        len(sentences) or len(raw_sentences), max_sentences, strict_max=strict_max
    )
 
    sim = build_similarity_matrix(sentences)
    scores = pagerank(sim)
    ranked_pairs = pick_top_sentences(sentences, scores, max_sentences) if sentences else []
 
    verdict_sents = extract_verdict_sentences(raw_sentences)
 
    sections = {
        "FACTS": [],
        "ISSUES": [],
        "PROCEDURAL HISTORY": [],
        "DECISION / VERDICT": [],
        "OTHER": [],
    }
 
    if verdict_sents:
        sections["DECISION / VERDICT"].append(verdict_sents[0])
    else:
        # Fallback: last sentence in the document with any verdict-adjacent
        # word, skipping procedural noise.
        for s in reversed(raw_sentences):
            sl = s.lower()
            has_verdict_word = (
                "decree" in sl or "dismissed" in sl or "allowed" in sl
                or "acquitted" in sl or "convicted" in sl
            )
            is_noise = any(p in sl for p in PROCEDURAL_NOISE)
            if has_verdict_word and not is_noise:
                sections["DECISION / VERDICT"].append(s)
                break
 
    already_included = set(sections["DECISION / VERDICT"])
    ranked_pairs_ordered = sorted(ranked_pairs, key=lambda x: x[0])
 
    for idx, sent in ranked_pairs_ordered:
        if sent in already_included:
            continue
        cat = classify_sentence(sent)
        if cat == "DECISION / VERDICT" and sections["DECISION / VERDICT"]:
            continue
        sections[cat].append(sent)
        already_included.add(sent)
 
    limits = {
        "FACTS": 2,
        "ISSUES": 1,
        "PROCEDURAL HISTORY": 2,
        "DECISION / VERDICT": 1,
        "OTHER": 2,
    }
 
    output = []
    for title, sents in sections.items():
        capped = sents[:limits[title]]
        if capped:
            output.append(f"{title}:\n")
            for s in capped:
                output.append(f"- {s}\n")
            output.append("\n")
 
    final_summary = "".join(output).strip()
    actual_count = sum(min(len(v), limits[k]) for k, v in sections.items())
 
    return SummaryResult(
        summary=final_summary,
        total_sentences=len(raw_sentences),
        sentences_used=actual_count,
    )
 
 
 
def extract_text_from_pdf(pdf_source: Union[str, IO[bytes]]) -> str:
    """Extract text from a PDF using pdfplumber.
 
    Parameters
    ----------
    pdf_source:
        Either a filesystem path (str) to a PDF, or a file-like/byte-stream
        object -- e.g. Flask's `request.files['file']` (a Werkzeug
        FileStorage, which behaves like a file object), an open `BytesIO`,
        or a file opened in 'rb' mode.
 
    Returns
    -------
    str
        The concatenated text of all pages, separated by newlines.
 
    Raises
    ------
    PDFExtractionError
        If pdfplumber isn't installed, the PDF can't be opened (corrupted,
        password-protected, not a valid PDF), or it contains no
        extractable text layer at all -- typical of a scanned/image-only
        PDF, which would need OCR (not handled here).
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise PDFExtractionError(
            "pdfplumber is required for PDF support. Install with: "
            "pip install pdfplumber"
        ) from exc
 
    pages_text: List[str] = []
    try:
        with pdfplumber.open(pdf_source) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
                else:
                    logger.debug("Page %d had no extractable text.", page_num)
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        raise PDFExtractionError(f"Failed to read PDF ({detail}). It may be corrupted or password-protected.") from exc
 
    full_text = "\n".join(pages_text).strip()
 
    if not full_text:
        raise PDFExtractionError(
            "No extractable text found in the PDF. This usually means it's "
            "a scanned image with no text layer (would require OCR, e.g. "
            "pytesseract, which this module does not include)."
        )
 
    return full_text
 
 
def summarize_pdf(
    pdf_source: Union[str, IO[bytes]],
    max_sentences: int = 8,
    strict_max: bool = True,
) -> SummaryResult:
    """Extract text from a PDF and summarize it in one call.
 
    Raises
    ------
    PDFExtractionError
        See `extract_text_from_pdf`.
    EmptyDocumentError, TypeError, ValueError
        See `summarize_text`.
    """
    text = extract_text_from_pdf(pdf_source)
    return summarize_text(text, max_sentences=max_sentences, strict_max=strict_max)
 