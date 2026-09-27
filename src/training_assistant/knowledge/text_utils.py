"""Shared text normalisation used by both the TF-IDF embedder and BM25 scoring.

Keeping one tokeniser means the dense and lexical halves of hybrid retrieval see
the same view of the text, which makes their scores comparable.
"""

from __future__ import annotations

import re
from functools import lru_cache

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")

#: Very small stop list. Deliberately conservative: words like "not", "no" and
#: "who" carry real meaning in a policy corpus and are kept.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the this that these those
    is are was were be been being am
    do does did doing done
    have has had having
    of in on at to for from by with within into onto about as
    and or but if then than so because while
    i me my we our you your it its they them their he she his her
    there here where when
    will would shall should can could may might must
    up down out over under again further once
    all any both each few more most other some such only own same
    s t just now very

    without through throughout during before after between against
    upon per via whether either neither however therefore thus hence
    otherwise instead rather already along across behind beyond
    toward towards regarding concerning including
    """.split()
)

#: Question words and filler that add noise to retrieval queries.
#:
#: Note: "find" is deliberately absent. It would stem together with "findings",
#: which is a meaningful corpus term ("audit findings", "security findings").
QUERY_NOISE: frozenset[str] = frozenset(
    """
    what which who whom whose how why
    tell explain describe give show help please
    mean means meaning
    need needs needed want wants
    let know
    """.split()
)


#: Plural rules, applied first. Each entry is (suffix, replacement, min_stem):
#: the rule fires only when the resulting stem is at least ``min_stem`` long.
#: There is deliberately no blanket "es" -> "" rule. Sibilant plurals are
#: handled explicitly and everything else drops only the final "s", so "offices"
#: reduces to "office" and matches the singular. A blanket "es" rule turned
#: "offices" into "offic" while leaving "office" untouched, and the two never met.
_PLURAL_RULES: tuple[tuple[str, str, int], ...] = (
    ("ies", "y", 4),
    ("sses", "ss", 4),
    ("ches", "ch", 4),
    ("shes", "sh", 4),
    ("xes", "x", 3),
    ("s", "", 3),
)

#: Derivational rules, applied *after* plural normalisation so that a plural
#: still reaches them: "bookings" -> "booking" -> "book" matches "booking".
#: The "-ation" family precedes the past-tense rule so "locations" and "located"
#: both reduce to "locat"; otherwise they become "location" and "locat".
_DERIVATIONAL_RULES: tuple[tuple[str, str, int], ...] = (
    ("ation", "at", 5),
    ("ate", "at", 5),
    ("ing", "", 4),
    ("edly", "", 4),
    ("ely", "", 4),
    ("ers", "", 4),
    ("er", "", 4),
    ("ed", "", 4),
    ("ly", "", 4),
)


def _apply(token: str, rules: tuple[tuple[str, str, int], ...]) -> str:
    """Apply the first matching rule whose stem is long enough."""
    for suffix, replacement, min_stem in rules:
        if not token.endswith(suffix):
            continue
        stem = token[: -len(suffix)] + replacement
        if len(stem) >= min_stem:
            return stem
    return token


#: Irregular forms that no suffix rule can reach. Kept deliberately short: only
#: verbs that actually appear in employee questions about this corpus. Without
#: "paid" -> "pay", "when do I get paid" shares no term with "pay date".
_IRREGULAR: dict[str, str] = {
    "paid": "pay",
    "paying": "pay",
    "pays": "pay",
    "got": "get",
    "gets": "get",
    "getting": "get",
    "gave": "give",
    "given": "give",
    "took": "take",
    "taken": "take",
    "made": "make",
    "told": "tell",
    "sold": "sell",
    "left": "leave",
    "ran": "run",
    "began": "begin",
    "chose": "choose",
    "held": "hold",
    "kept": "keep",
    "sent": "send",
    "spent": "spend",
    "lost": "lose",
    "met": "meet",
    "built": "build",
    "bought": "buy",
    "brought": "bring",
    "caught": "catch",
    "taught": "teach",
    "thought": "think",
    "wrote": "write",
    "written": "write",
}


@lru_cache(maxsize=8192)
def _stem(token: str) -> str:
    """Crude two-stage suffix stripper.

    Not a linguistic stemmer. Its only job is to make "policies"/"policy",
    "offices"/"office" and "locations"/"located" collide, which matters a great
    deal on a corpus this small: a query and the passage that answers it often
    use different inflections of the same word.

    Stage one normalises plurals, stage two strips derivational endings. Running
    them in that order (rather than as one rule list) is what lets a plural
    gerund such as "bookings" reach the "-ing" rule and match "booking".
    """
    irregular = _IRREGULAR.get(token)
    if irregular is not None:
        return irregular

    if len(token) <= 3:
        return token

    # "process", "class", "address", "business": no plural rule applies, and
    # stripping the final "s" would make the singular stem differently.
    if not token.endswith("ss"):
        token = _apply(token, _PLURAL_RULES)

    if len(token) <= 3 or token.endswith("ss"):
        return token

    return _apply(token, _DERIVATIONAL_RULES)


@lru_cache(maxsize=1)
def _noise_stems() -> frozenset[str]:
    """Stems of the query-noise words.

    The noise list has to be matched *after* stemming as well as before it.
    Checking the raw token only lets inflected forms slip through: "telling"
    is not in the list, survives filtering, then stems to "tell" and collides
    with corpus phrases such as "tell your manager". That is not hypothetical --
    it pushed an onboarding-checklist passage above the Code of Conduct on the
    question "how do I report harassment without telling my manager?".
    """
    return frozenset(_stem(word) for word in QUERY_NOISE)


@lru_cache(maxsize=1)
def _stopword_stems() -> frozenset[str]:
    """Stems of the stop list, for the same reason as :func:`_noise_stems`."""
    return frozenset(_stem(word) for word in STOPWORDS)


def tokenize(text: str, *, drop_query_noise: bool = False) -> list[str]:
    """Lowercase, split, drop stopwords and stem.

    Args:
        text: Raw input text.
        drop_query_noise: Additionally remove interrogatives and filler verbs.
            Enable this for queries, not for documents.
    """
    noise = _noise_stems() if drop_query_noise else frozenset()
    stopword_stems = _stopword_stems()

    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS or (drop_query_noise and raw in QUERY_NOISE):
            continue

        stem = _stem(raw)
        if stem in stopword_stems or stem in noise:
            continue

        tokens.append(stem)
    return tokens
