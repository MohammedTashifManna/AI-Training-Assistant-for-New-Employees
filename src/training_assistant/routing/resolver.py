"""Two-stage route resolution: keyword rules first, then a retrieval probe.

Why a second stage exists
-------------------------
The rule-based classifier scores 100% on the labelled question set and **48%**
on a held-out set of casual paraphrases. Nearly every failure looked the same:
no keyword matched, so the question fell through to the general path.

::

    "i paid for a taxi myself, how do i get that money back"  -> general
    "if i quit how long do i have to stick around"            -> general
    "who runs this place"                                     -> general

The instinct is to add "taxi", "quit" and "runs this place" to the keyword
lists. That is a treadmill: it fixes the sentences you have already seen and
teaches you nothing about the ones you have not.

The structural fix is to stop treating a keyword miss as evidence of an
out-of-scope question. A vocabulary gap in the *router* says nothing about
whether the *corpus* can answer the question, and the corpus is right there. So
when the rules are unsure, this module asks each knowledge collection how well
it can answer, and routes to the collection that answers best. Retrieval already
does robust term weighting over the real document vocabulary, which is exactly
the signal the keyword lists lack.

Genuinely out-of-scope questions are protected by a relevance floor that is
deliberately higher than the one used for normal retrieval: a question about
vector databases or the weather scores below it everywhere and correctly stays
on the general path.

This keeps the deterministic path honest and cheap: two thresholds, no growing
keyword list, and one extra retrieval per uncertain question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

from .classifier import CONFIDENCE_FLOOR, RouteDecision, classify

logger = logging.getLogger(__name__)

#: Knowledge domains a retrieval probe can choose between.
PROBE_DOMAINS: Final[tuple[str, ...]] = ("company", "role", "admin")

#: A probe must reach this top-1 hybrid retrieval score to claim the question.
#: Higher than the retriever's own ``min_score``: a passage that is merely
#: *not irrelevant* is not enough to override the router's uncertainty.
PROBE_MIN_SCORE: Final[float] = 0.16

#: The winning domain must beat the runner-up by this margin. Without it, a
#: question that all three collections answer equally weakly gets assigned to
#: whichever one happens to be marginally ahead.
PROBE_MIN_MARGIN: Final[float] = 0.015

#: Rule-based decisions at or above this confidence are trusted outright and
#: skip the probe entirely.
TRUST_RULES_ABOVE: Final[float] = 0.5


@dataclass(slots=True)
class ResolvedRoute:
    """Final routing decision, with the evidence behind it."""

    route: str
    reason: str
    decided_by: str  # "rules" | "retrieval_probe" | "fallback"
    rule_decision: RouteDecision | None = None
    probe_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "reason": self.reason,
            "decided_by": self.decided_by,
            "probe_scores": {k: round(v, 4) for k, v in self.probe_scores.items()},
            "rule_route": self.rule_decision.route if self.rule_decision else "",
            "rule_confidence": (
                round(self.rule_decision.confidence, 4) if self.rule_decision else 0.0
            ),
        }


def _probe_domains(question: str) -> dict[str, float]:
    """Best retrieval score each knowledge collection can achieve.

    Returns an empty mapping when the index is unavailable, so routing degrades
    to the rule-based decision rather than failing.
    """
    try:
        from ..knowledge.retriever import get_retriever

        retriever = get_retriever()
    except Exception:
        logger.debug("Retrieval probe unavailable; using rules alone.", exc_info=True)
        return {}

    scores: dict[str, float] = {}
    for domain in PROBE_DOMAINS:
        try:
            # min_score=0 so the raw best score is visible; the probe applies its
            # own, stricter threshold below.
            results = retriever.search(domain, question, top_k=1, min_score=0.0)
        except Exception:
            logger.debug("Probe failed for domain %s", domain, exc_info=True)
            continue
        scores[domain] = results[0].score if results else 0.0

    return scores


def resolve_route(question: str) -> ResolvedRoute:
    """Decide which path should answer ``question``.

    Stage one is the keyword classifier. Stage two, used only when stage one is
    unsure, asks the knowledge collections directly.
    """
    decision = classify(question)

    # Stage one is trusted in two situations.
    #
    # (a) It picked a knowledge route with real confidence.
    confident_knowledge_route = (
        decision.route != "general" and decision.confidence >= TRUST_RULES_ABOVE
    )

    # (b) It picked the general route on *positive* evidence: the question
    #     contained generic-topic signals such as "in Python", "SQL join",
    #     "tips for" or "can you recommend". That is a different situation from
    #     the general route being reached because nothing matched at all, and
    #     conflating the two is what makes a retrieval probe harmful. Every
    #     knowledge base can return *something* for "explain SQL joins", so
    #     probing an already-well-explained general question drags it into a
    #     knowledge route and produces a confidently irrelevant answer.
    general_on_positive_evidence = decision.route == "general" and bool(
        decision.matched.get("general")
    )

    if confident_knowledge_route or general_on_positive_evidence:
        return ResolvedRoute(
            route=decision.route,
            reason=decision.reason,
            decided_by="rules",
            rule_decision=decision,
        )

    scores = _probe_domains(question)
    if not scores:
        return ResolvedRoute(
            route=decision.route,
            reason=decision.reason,
            decided_by="rules",
            rule_decision=decision,
        )

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_domain, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - runner_up

    if best_score >= PROBE_MIN_SCORE and margin >= PROBE_MIN_MARGIN:
        # If the rules already picked a knowledge route, only override it when
        # the probe disagrees clearly; a weak rule signal plus an agreeing probe
        # is still the rules' decision.
        if decision.route == best_domain:
            reason = (
                f"Matched {best_domain} signals weakly, and the {best_domain} "
                f"knowledge base answered best (relevance {best_score:.2f})."
            )
        else:
            reason = (
                f"Keyword signals were inconclusive, so the question was put to "
                f"each knowledge base: {best_domain} answered best "
                f"(relevance {best_score:.2f} vs {runner_up:.2f} for the "
                f"next best)."
            )
        logger.info(
            "Retrieval probe routed %r to %s (scores=%s)",
            question[:60],
            best_domain,
            {k: round(v, 3) for k, v in scores.items()},
        )
        return ResolvedRoute(
            route=best_domain,
            reason=reason,
            decided_by="retrieval_probe",
            rule_decision=decision,
            probe_scores=scores,
        )

    reason = (
        "No knowledge base could answer this well "
        f"(best relevance {best_score:.2f}, below the {PROBE_MIN_SCORE:.2f} "
        "threshold), so it was handled as a general question."
    )
    logger.info(
        "Retrieval probe found nothing for %r (scores=%s)",
        question[:60],
        {k: round(v, 3) for k, v in scores.items()},
    )
    return ResolvedRoute(
        route="general",
        reason=reason,
        decided_by="fallback",
        rule_decision=decision,
        probe_scores=scores,
    )


__all__ = [
    "CONFIDENCE_FLOOR",
    "PROBE_MIN_MARGIN",
    "PROBE_MIN_SCORE",
    "ResolvedRoute",
    "TRUST_RULES_ABOVE",
    "resolve_route",
]
