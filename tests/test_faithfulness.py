"""Faithfulness and hallucination tests, using an LLM-as-a-Judge pattern.

The question this suite answers
-------------------------------
Not "is the answer good?" but "does the answer assert anything the source
documents do not support?" For an onboarding assistant those are very different
questions. A terse answer that quotes the leave policy correctly is a success. A
fluent answer that invents an expense cap is a failure that costs an employee
money, and it will read *better* than the correct one.

What counts as ungrounded
-------------------------
Answers are judged against the **union of two contexts**:

1. the ground-truth ``reference_chunks`` in ``eval_dataset.json``, copied
   verbatim from the corpus, and
2. the passages the retriever actually returned for that question.

Using the union is deliberate. If the retriever fetched a different but equally
valid section than the one the dataset names, the answer is still grounded, and
scoring it as a hallucination would conflate a retrieval-ranking difference with
a fabrication. Whether retrieval found the *expected* document is a separate
question, tested separately in :func:`test_retrieval_reaches_expected_documents`.

Two judges
----------
:class:`GeminiJudge`
    The LLM-as-a-Judge. Decomposes the answer into atomic claims and rules on
    each one against the context, returning structured JSON. Used when an API
    key is available.

:class:`GroundingJudge`
    A deterministic checker used when no key is present, so the suite still
    runs. **It is not an LLM judge** and does not attempt semantic entailment.
    It targets the failure mode that actually matters here: fabricated
    *specifics*. Every figure, monetary amount, duration, percentage, channel
    name and email address in the answer must appear in the context. That
    catches invented caps, wrong notice periods and made-up Slack channels,
    which is the bulk of the real risk, while saying nothing about subtler
    misparaphrase.

Because a test that can never fail is worse than no test,
:func:`test_judge_detects_an_injected_hallucination` feeds the active judge a
deliberately corrupted answer and asserts it is caught.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest

from conftest import publish_scorecard

pytestmark = pytest.mark.faithfulness

# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #

#: Share of grounded answers required across the retrieval routes.
MIN_FAITHFULNESS_RATE = 0.95

#: Fabricated specifics are not tolerated at all: one invented expense cap is
#: one too many.
MAX_CRITICAL_UNGROUNDED = 0

#: Every bracketed citation in an answer must resolve to a retrieved passage.
MAX_DANGLING_CITATIONS = 0


# --------------------------------------------------------------------------- #
# Fact extraction
# --------------------------------------------------------------------------- #

_MARKDOWN_NOISE = re.compile(r"[*_`>#]+")
_WHITESPACE = re.compile(r"\s+")

#: Currency amounts: "INR 9,000", "USD 250", "$0.62", "Rs. 2000".
_CURRENCY = re.compile(
    r"\b(?:INR|USD|EUR|GBP|Rs\.?)\s*([\d,]+(?:\.\d+)?)|([$€£])\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

#: Quantities with a unit: "22 working days", "26 weeks", "118%", "14 per km".
_QUANTITY = re.compile(
    r"\b([\d,]+(?:\.\d+)?)\s*"
    r"(%|percent|working\s+days?|business\s+days?|calendar\s+days?|days?|weeks?|"
    r"months?|years?|hours?|minutes?|kilometres?|kilometers?|km|miles?)\b",
    re.IGNORECASE,
)

#: Any remaining standalone number, including grade labels such as "L5".
_BARE_NUMBER = re.compile(r"\b(?:L)?(\d[\d,]*(?:\.\d+)?)\b", re.IGNORECASE)

#: Slack channels and email addresses are specifics that get invented easily.
_CHANNEL = re.compile(r"#[a-z0-9][a-z0-9\-_]{2,}", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")

#: Numbers that carry no policy meaning and would only create noise.
_IGNORED_NUMBERS = {"1", "2", "3", "4", "5", "0"}


def _plain(text: str) -> str:
    """Strip Markdown emphasis and collapse whitespace for matching."""
    return _WHITESPACE.sub(" ", _MARKDOWN_NOISE.sub("", text)).strip()


def _normalise_number(value: str) -> str:
    return value.replace(",", "").rstrip(".").lower()


def extract_specifics(text: str) -> dict[str, set[str]]:
    """Pull checkable specifics out of a piece of text.

    Returns a mapping of kind to normalised values. Kinds are kept apart so a
    monetary amount is never considered matched by an unrelated bare number that
    happens to share digits.
    """
    plain = _plain(text)

    money: set[str] = set()
    for match in _CURRENCY.finditer(plain):
        amount = match.group(1) or match.group(3)
        if amount:
            money.add(_normalise_number(amount))

    quantities: set[str] = set()
    for match in _QUANTITY.finditer(plain):
        number = _normalise_number(match.group(1))
        unit = match.group(2).lower()
        unit = _WHITESPACE.sub(" ", unit)
        # Normalise plurals and synonyms so "22 days" matches "22 working days"
        # only via the number, not by pretending the units are identical.
        unit = unit.rstrip("s") if unit not in {"%", "percent"} else "%"
        quantities.add(f"{number} {unit}".strip())

    numbers: set[str] = set()
    for match in _BARE_NUMBER.finditer(plain):
        value = _normalise_number(match.group(1))
        if value not in _IGNORED_NUMBERS:
            numbers.add(value)

    return {
        "money": money,
        "quantity": quantities,
        "number": numbers,
        "channel": {c.lower() for c in _CHANNEL.findall(plain)},
        "email": {e.lower() for e in _EMAIL.findall(plain)},
    }


def split_claims(answer: str) -> list[str]:
    """Break an answer into atomic claims.

    Bullets and table-derived lines are claims in their own right; prose is split
    on sentence boundaries. Headings, source lists and the offline-mode notice
    are metadata rather than assertions about policy, so they are dropped.
    """
    claims: list[str] = []
    in_sources = False

    for raw in answer.splitlines():
        line = raw.strip()
        if not line:
            continue

        lowered = _plain(line).lower()
        if lowered.startswith("sources"):
            in_sources = True
            continue
        if in_sources and (line.startswith("-") or line.startswith("[")):
            continue
        if lowered.startswith("offline mode:"):
            continue
        if lowered.startswith("here is what the onboarding documents say"):
            continue
        if line.startswith("#"):
            continue

        body = re.sub(r"^[-*+]\s+", "", line)
        body = re.sub(r"^\d+\.\s+", "", body)
        if not body.strip():
            continue

        if body.startswith("|") or len(body) < 90:
            claims.append(body.strip())
        else:
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", body):
                if sentence.strip():
                    claims.append(sentence.strip())

    return claims


# --------------------------------------------------------------------------- #
# Verdict types
# --------------------------------------------------------------------------- #


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    severity: str = "none"  # "critical" | "minor" | "none"
    reason: str = ""
    evidence: str = ""

    @property
    def is_critical(self) -> bool:
        return not self.supported and self.severity == "critical"


@dataclass
class JudgeVerdict:
    judge_name: str
    claims: list[ClaimVerdict] = field(default_factory=list)
    error: str = ""

    @property
    def ungrounded(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if not c.supported]

    @property
    def critical(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if c.is_critical]

    @property
    def grounded(self) -> bool:
        return not self.ungrounded

    @property
    def support_rate(self) -> float:
        if not self.claims:
            return 1.0
        return sum(1 for c in self.claims if c.supported) / len(self.claims)

    def describe(self) -> str:
        if self.grounded:
            return f"{len(self.claims)} claim(s), all grounded"
        parts = [f"{len(self.ungrounded)}/{len(self.claims)} claim(s) ungrounded"]
        for claim in self.ungrounded[:4]:
            parts.append(f"    [{claim.severity}] {claim.claim[:90]}")
            if claim.reason:
                parts.append(f"        reason: {claim.reason[:110]}")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Judges
# --------------------------------------------------------------------------- #


class Judge(ABC):
    name = "abstract"

    @abstractmethod
    def evaluate(self, question: str, answer: str, context: Sequence[str]) -> JudgeVerdict:
        """Rule on whether ``answer`` is supported by ``context``."""

    @property
    def is_llm(self) -> bool:
        return False


class GroundingJudge(Judge):
    """Deterministic grounding checker. Not an LLM judge.

    A claim fails when it contains a specific -- a figure, monetary amount,
    duration, percentage, Slack channel or email address -- that does not appear
    anywhere in the supplied context. Those are marked ``critical``, because an
    invented number is the failure mode with real consequences.

    What it deliberately does not do is judge semantic entailment. A claim that
    reuses only grounded numbers but misstates their meaning will pass here. That
    limit is the reason the Gemini judge exists.
    """

    name = "grounding-checker"

    def evaluate(self, question: str, answer: str, context: Sequence[str]) -> JudgeVerdict:
        context_specifics = extract_specifics("\n".join(context))
        verdict = JudgeVerdict(judge_name=self.name)

        for claim in split_claims(answer):
            claim_specifics = extract_specifics(claim)
            missing: list[str] = []

            for kind in ("money", "quantity", "number", "channel", "email"):
                for value in sorted(claim_specifics[kind] - context_specifics[kind]):
                    # A quantity is acceptable when its bare number is present in
                    # the context; unit phrasing differs harmlessly between a
                    # table cell and a sentence.
                    if kind == "quantity":
                        number = value.split(" ", 1)[0]
                        if number in context_specifics["number"] or number in context_specifics["money"]:
                            continue
                    if kind == "number" and value in context_specifics["money"]:
                        continue
                    if kind == "money" and value in context_specifics["number"]:
                        continue
                    missing.append(f"{kind}:{value}")

            if missing:
                verdict.claims.append(
                    ClaimVerdict(
                        claim=claim,
                        supported=False,
                        severity="critical",
                        reason=(
                            "specifics absent from the source context: "
                            + ", ".join(missing[:6])
                        ),
                    )
                )
            else:
                verdict.claims.append(ClaimVerdict(claim=claim, supported=True))

        return verdict


_JUDGE_PROMPT = """\
You are a strict grounding auditor for an employee onboarding assistant. Your
job is to detect claims that the source context does not support.

Rules:
- Break the ANSWER into atomic factual claims. Ignore pleasantries, offers to
  help, headings, and lists of source citations.
- For each claim decide whether the CONTEXT supports it.
- A claim is supported only if the context states it or directly entails it.
  Plausibility is not support. General world knowledge is not support.
- Any figure, monetary amount, deadline, duration, percentage, job title, person
  name, system name, Slack channel or email address that does not appear in the
  context is UNSUPPORTED with severity "critical".
- A claim that is broadly consistent with the context but adds unsupported
  qualification is UNSUPPORTED with severity "minor".
- A refusal such as "the documents do not cover this" is SUPPORTED.
- Quote the exact supporting span from the context in "evidence" when supported,
  and leave it empty when not.

Return JSON only, matching this shape:
{"claims": [{"claim": "...", "supported": true, "severity": "none",
             "evidence": "...", "reason": "..."}]}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


class GeminiJudge(Judge):
    """LLM-as-a-Judge backed by Gemini, returning structured claim verdicts."""

    name = "gemini-judge"

    def __init__(self, model: str, api_key: str = "") -> None:
        self.model = model
        self._api_key = api_key
        self._client: Any = None

    @property
    def is_llm(self) -> bool:
        return True

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = (
                genai.Client(api_key=self._api_key) if self._api_key else genai.Client()
            )
        return self._client

    def evaluate(self, question: str, answer: str, context: Sequence[str]) -> JudgeVerdict:
        from google.genai import types

        prompt = (
            _JUDGE_PROMPT.replace("{question}", question)
            .replace("{context}", "\n\n---\n\n".join(context) or "(no context)")
            .replace("{answer}", answer)
        )

        try:
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            payload = json.loads(response.text or "{}")
        except Exception as exc:  # pragma: no cover - network dependent
            return JudgeVerdict(judge_name=self.name, error=f"{type(exc).__name__}: {exc}")

        verdict = JudgeVerdict(judge_name=self.name)
        for item in payload.get("claims", []):
            if not isinstance(item, dict):
                continue
            supported = bool(item.get("supported"))
            verdict.claims.append(
                ClaimVerdict(
                    claim=str(item.get("claim", ""))[:400],
                    supported=supported,
                    severity="none" if supported else str(item.get("severity") or "minor"),
                    reason=str(item.get("reason", ""))[:300],
                    evidence=str(item.get("evidence", ""))[:300],
                )
            )
        return verdict


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def judge(judge_choice, settings) -> Judge:
    """Select the judge, honouring ``--judge``."""
    api_key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )

    if judge_choice == "grounding":
        return GroundingJudge()

    if judge_choice == "gemini":
        if not api_key:
            pytest.fail(
                "--judge=gemini requires GOOGLE_API_KEY or GEMINI_API_KEY to be set."
            )
        return GeminiJudge(model=settings.model, api_key=api_key)

    # auto
    if api_key:
        return GeminiJudge(model=settings.model, api_key=api_key)
    return GroundingJudge()


def _contexts(question: dict[str, Any], answer) -> list[str]:
    """Ground-truth reference chunks plus what retrieval actually returned."""
    contexts = [chunk["text"] for chunk in question.get("reference_chunks", [])]
    contexts.extend(source.text for source in answer.sources if source.text)
    return contexts


def _load_dataset() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "eval_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieval_cases():
    return [
        pytest.param(q["id"], id=q["id"])
        for q in _load_dataset()["routing_questions"]
        if q["expects_retrieval"]
    ]


def _fallback_cases():
    return [
        pytest.param(q["id"], id=q["id"])
        for q in _load_dataset()["routing_questions"]
        if not q["expects_retrieval"]
    ]


def _probe_cases():
    return [pytest.param(p["id"], id=p["id"]) for p in _load_dataset()["hallucination_probes"]]


@pytest.fixture(scope="module")
def questions_by_id(routing_questions) -> dict[str, dict[str, Any]]:
    return {q["id"]: q for q in routing_questions}


@pytest.fixture(scope="module")
def probes_by_id(hallucination_probes) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in hallucination_probes}


# --------------------------------------------------------------------------- #
# Dataset integrity
# --------------------------------------------------------------------------- #


def test_reference_chunks_match_corpus(routing_questions, retriever):
    """Ground truth must still match the corpus, verbatim.

    The dataset stores reference text copied out of ``data/``. If a document is
    edited and the dataset is not, every faithfulness result silently becomes
    meaningless: answers would be judged against text the system can no longer
    retrieve. This test is what makes the stored copies safe to rely on.
    """
    corpus = {chunk.citation: chunk.text for chunk in retriever.store.iter_chunks()}
    problems: list[str] = []

    for question in routing_questions:
        for reference in question["reference_chunks"]:
            citation = reference["citation"]
            if citation not in corpus:
                problems.append(
                    f"{question['id']}: citation no longer exists: {citation!r}"
                )
            elif corpus[citation].strip() != reference["text"].strip():
                problems.append(
                    f"{question['id']}: text drifted for {citation!r}"
                )

    assert not problems, (
        "The evaluation dataset is out of sync with the corpus:\n  "
        + "\n  ".join(problems)
        + "\n\nRe-extract the affected reference_chunks from data/."
    )


# --------------------------------------------------------------------------- #
# Judge self-test
# --------------------------------------------------------------------------- #


def test_judge_detects_an_injected_hallucination(judge, questions_by_id, pipeline_results):
    """The judge must fail an answer we have deliberately corrupted.

    Without this, a judge that returns "grounded" unconditionally would make the
    entire suite pass while detecting nothing. This is the control.
    """
    question = questions_by_id["R12"]  # nightly hotel cap for Mumbai
    answer = pipeline_results["R12"]
    context = _contexts(question, answer)

    fabricated = (
        "The nightly hotel cap for Mumbai is INR 47,500 and claims must be "
        "submitted within 92 days to #ask-nonexistent-channel."
    )

    verdict = judge.evaluate(question["question"], fabricated, context)
    if verdict.error:
        pytest.skip(f"Judge unavailable: {verdict.error}")

    assert not verdict.grounded, (
        f"{judge.name} passed a fabricated answer containing INR 47,500, a "
        "92-day window and an invented Slack channel. The judge is not "
        "detecting hallucination, so no other result in this file can be "
        "trusted."
    )
    assert verdict.critical, (
        f"{judge.name} flagged the fabrication but not as critical: "
        f"{verdict.describe()}"
    )


def test_judge_accepts_a_faithful_answer(judge, questions_by_id, pipeline_results):
    """The control in the other direction: no false alarm on grounded text.

    A judge that fails everything would also make the suite meaningless, just
    noisily rather than silently.
    """
    question = questions_by_id["R12"]
    answer = pipeline_results["R12"]
    context = _contexts(question, answer)

    faithful = (
        "The nightly cap for Mumbai and Delhi NCR is INR 9,000, excluding taxes."
    )

    verdict = judge.evaluate(question["question"], faithful, context)
    if verdict.error:
        pytest.skip(f"Judge unavailable: {verdict.error}")

    assert verdict.grounded, (
        f"{judge.name} rejected an answer that quotes the source correctly:\n"
        f"{verdict.describe()}"
    )


# --------------------------------------------------------------------------- #
# Per-question faithfulness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("qid", _retrieval_cases())
def test_answer_has_no_ungrounded_claims(qid, judge, questions_by_id, pipeline_results):
    """Every claim in a retrieval-backed answer must trace to the sources."""
    question = questions_by_id[qid]
    answer = pipeline_results[qid]
    assert not answer.error, f"{qid} raised: {answer.error}"

    context = _contexts(question, answer)
    assert context, f"{qid} has no context to judge against"

    verdict = judge.evaluate(question["question"], answer.answer, context)
    if verdict.error:
        pytest.skip(f"Judge unavailable: {verdict.error}")

    assert len(verdict.critical) <= MAX_CRITICAL_UNGROUNDED, (
        f"{qid} contains fabricated specifics.\n"
        f"  question: {question['question']}\n"
        f"  judge   : {verdict.judge_name}\n"
        f"  {verdict.describe()}\n"
        f"  answer  : {answer.answer[:400]}"
    )


@pytest.mark.parametrize("qid", _retrieval_cases())
def test_answer_contains_the_required_facts(qid, questions_by_id, pipeline_results):
    """Grounding is necessary but not sufficient: the answer must be useful.

    An answer can be perfectly faithful and still miss the point by quoting a
    neighbouring paragraph. Each dataset entry names the facts a correct answer
    has to carry, in two forms:

    ``must_contain_facts``
        Every one of these must appear. Used for single-valued answers such as a
        cap or a notice period.
    ``must_contain_any``
        At least one must appear. Used where several answers are equally correct,
        for instance naming any of the four public reference customers.
    """
    question = questions_by_id[qid]
    required = question.get("must_contain_facts") or []
    any_of = question.get("must_contain_any") or []

    if not required and not any_of:
        pytest.skip(f"{qid} declares no required facts")

    answer_text = _plain(pipeline_results[qid].answer).lower()

    missing = [fact for fact in required if _plain(fact).lower() not in answer_text]
    assert not missing, (
        f"{qid} omitted required fact(s) {missing}.\n"
        f"  question: {question['question']}\n"
        f"  answer  : {pipeline_results[qid].answer[:400]}"
    )

    if any_of:
        matched = [fact for fact in any_of if _plain(fact).lower() in answer_text]
        assert matched, (
            f"{qid} contained none of {any_of}; at least one is required.\n"
            f"  question: {question['question']}\n"
            f"  answer  : {pipeline_results[qid].answer[:400]}"
        )


@pytest.mark.parametrize("qid", _retrieval_cases())
def test_citations_resolve_to_retrieved_passages(qid, pipeline_results):
    """A bracketed citation must point at a passage that was actually retrieved.

    A dangling ``[7]`` is a small but corrosive failure: it looks like evidence
    and cannot be checked.
    """
    answer = pipeline_results[qid]
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer.answer)}
    if not cited:
        pytest.skip(f"{qid} answer carries no inline citations")

    available = {source.number for source in answer.sources}
    dangling = sorted(cited - available)

    assert len(dangling) <= MAX_DANGLING_CITATIONS, (
        f"{qid} cites passage(s) {dangling} that were never retrieved. "
        f"Available: {sorted(available)}"
    )


@pytest.mark.parametrize("qid", _retrieval_cases())
def test_retrieval_reaches_expected_documents(qid, questions_by_id, pipeline_results):
    """Retrieval quality, kept separate from faithfulness.

    An answer grounded in the *wrong* document would pass the grounding tests, so
    this asserts that at least one expected source document was actually reached.
    """
    question = questions_by_id[qid]
    expected = set(question["expected_doc_ids"])
    retrieved = {source.doc_id for source in pipeline_results[qid].sources}

    assert expected & retrieved, (
        f"{qid} retrieved {sorted(retrieved)} but none of the expected "
        f"documents {sorted(expected)}.\n  question: {question['question']}"
    )


# --------------------------------------------------------------------------- #
# The fallback path must not assert company facts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("qid", _fallback_cases())
def test_fallback_answers_assert_no_company_specifics(qid, questions_by_id, pipeline_results):
    """The fallback agent has no retrieval tool, so it must not state policy.

    This is the path where a hallucinated Manna's fact is most likely and least
    detectable, because there are no sources on screen to contradict it.
    """
    answer = pipeline_results[qid]
    assert not answer.sources, f"{qid} took the fallback path but has sources"

    specifics = extract_specifics(answer.answer)
    money = specifics["money"]
    channels = specifics["channel"]

    assert not money, (
        f"{qid} took the no-retrieval fallback path yet quoted monetary "
        f"amount(s) {sorted(money)}. With no sources, any figure is invented.\n"
        f"  answer: {answer.answer[:300]}"
    )
    assert not channels, (
        f"{qid} invented internal channel(s) {sorted(channels)} on the "
        f"no-retrieval path.\n  answer: {answer.answer[:300]}"
    )


# --------------------------------------------------------------------------- #
# Adversarial probes: things that are not in the corpus at all
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pid", _probe_cases())
def test_absent_topics_are_not_invented(pid, probes_by_id, probe_results):
    """When the corpus has no answer, the assistant must not manufacture one.

    Each probe names a benefit or role that does not exist. Asserting the answer
    does not affirm it is a much stronger signal than asserting it said the
    words "I don't know", which a real system phrases many different ways.
    """
    probe = probes_by_id[pid]
    answer = probe_results[pid]
    assert not answer.error, f"{pid} raised: {answer.error}"

    text = _plain(answer.answer).lower()
    asserted = [
        claim for claim in probe["forbidden_claims"] if _plain(claim).lower() in text
    ]

    assert not asserted, (
        f"{pid} asserted {asserted}, which does not exist in the corpus.\n"
        f"  question : {probe['question']}\n"
        f"  rationale: {probe['rationale']}\n"
        f"  answer   : {answer.answer[:350]}"
    )


@pytest.mark.parametrize("pid", _probe_cases())
def test_probe_answers_stay_grounded(pid, judge, probes_by_id, probe_results):
    """Whatever a probe answer does say must still trace to retrieved sources."""
    probe = probes_by_id[pid]
    answer = probe_results[pid]

    context = [source.text for source in answer.sources if source.text]
    if not context:
        # Nothing retrieved: the only safe answer is one with no specifics.
        specifics = extract_specifics(answer.answer)
        invented = specifics["money"] | specifics["channel"] | specifics["email"]
        assert not invented, (
            f"{pid} retrieved nothing yet stated specifics {sorted(invented)}"
        )
        return

    verdict = judge.evaluate(probe["question"], answer.answer, context)
    if verdict.error:
        pytest.skip(f"Judge unavailable: {verdict.error}")

    assert len(verdict.critical) <= MAX_CRITICAL_UNGROUNDED, (
        f"{pid} produced fabricated specifics.\n"
        f"  question: {probe['question']}\n"
        f"  {verdict.describe()}"
    )


# --------------------------------------------------------------------------- #
# Aggregate scorecard
# --------------------------------------------------------------------------- #


def test_overall_faithfulness_rate(
    judge, routing_questions, pipeline_results, hallucination_probes, probe_results, request
):
    """Aggregate the per-answer verdicts into one reported rate."""
    rows: list[tuple[str, JudgeVerdict | None]] = []
    skipped = 0

    for question in routing_questions:
        if not question["expects_retrieval"]:
            continue
        answer = pipeline_results[question["id"]]
        verdict = judge.evaluate(
            question["question"], answer.answer, _contexts(question, answer)
        )
        if verdict.error:
            skipped += 1
            rows.append((question["id"], None))
            continue
        rows.append((question["id"], verdict))

    judged = [(qid, v) for qid, v in rows if v is not None]
    if not judged:
        pytest.skip("Judge unavailable for every question")

    grounded = sum(1 for _qid, v in judged if v.grounded)
    critical_total = sum(len(v.critical) for _qid, v in judged)
    claim_total = sum(len(v.claims) for _qid, v in judged)
    supported_total = sum(sum(1 for c in v.claims if c.supported) for _qid, v in judged)
    rate = grounded / len(judged)

    lines = [
        f"  judge                 : {judge.name}"
        + ("  (LLM-as-a-Judge)" if judge.is_llm else "  (deterministic, not an LLM)"),
        f"  answers judged        : {len(judged)}"
        + (f"  ({skipped} skipped)" if skipped else ""),
        f"  fully grounded        : {grounded}/{len(judged)} = {rate:.1%}",
        f"  claims checked        : {claim_total}",
        f"  claims supported      : {supported_total}/{claim_total}"
        + (f" = {supported_total / claim_total:.1%}" if claim_total else ""),
        f"  critical fabrications : {critical_total}",
        "",
        f"  probes (absent topics): {len(hallucination_probes)} checked, "
        f"{sum(1 for p in hallucination_probes if not probe_results[p['id']].sources)}"
        " answered with no sources",
    ]

    failures = [(qid, v) for qid, v in judged if not v.grounded]
    if failures:
        lines.append("")
        lines.append("  ungrounded answers:")
        for qid, verdict in failures:
            lines.append(f"    {qid}: {verdict.describe()}")

    if not judge.is_llm:
        lines.append("")
        lines.append(
            "  NOTE: the deterministic checker verifies that every figure, amount,"
        )
        lines.append(
            "  duration, channel and email in an answer appears in the source"
        )
        lines.append(
            "  context. It does not judge semantic entailment. Run with a Gemini"
        )
        lines.append("  key and --judge=gemini for full LLM-as-a-Judge evaluation.")

    publish_scorecard(request.config, "FAITHFULNESS", lines)

    assert critical_total <= MAX_CRITICAL_UNGROUNDED, (
        f"{critical_total} critical fabrication(s) detected.\n" + "\n".join(lines)
    )
    assert rate >= MIN_FAITHFULNESS_RATE, (
        f"Faithfulness rate {rate:.1%} is below {MIN_FAITHFULNESS_RATE:.0%}.\n"
        + "\n".join(lines)
    )
