from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.data_loader import Par4pcCase, PatentCandidate
from src.retrieval import RetrievalResult, rank_candidates_bm25, tokenize


PATENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "claim",
    "comprising",
    "computer",
    "configured",
    "each",
    "for",
    "from",
    "in",
    "is",
    "method",
    "more",
    "of",
    "on",
    "one",
    "or",
    "processor",
    "processors",
    "responsive",
    "said",
    "step",
    "system",
    "the",
    "thereof",
    "to",
    "wherein",
    "with",
}


@dataclass(frozen=True)
class ClaimLimitation:
    label: str
    text: str


@dataclass(frozen=True)
class EvidenceMatch:
    limitation_label: str
    limitation_text: str
    candidate_letter: str
    patent_id: str
    source: str
    evidence: str
    score: float
    verification: str = "not_verified"
    verification_reason: str = ""


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reason: str


def decompose_claim_heuristic(claim_text: str) -> list[ClaimLimitation]:
    normalized = " ".join(claim_text.split())
    body = normalized
    if " comprising:" in normalized:
        body = normalized.split(" comprising:", 1)[1]
    elif " comprising" in normalized:
        body = normalized.split(" comprising", 1)[1]

    pieces = [piece.strip(" ;.") for piece in re.split(r";\s*", body) if piece.strip(" ;.")]
    if len(pieces) <= 1:
        pieces = [piece.strip(" ;.") for piece in re.split(r",\s+(?=(?:wherein|receiving|generating|determining|responsive|analyzing|providing|storing|transmitting)\b)", body) if piece.strip(" ;.")]

    return [
        ClaimLimitation(label=f"L{i + 1}", text=piece)
        for i, piece in enumerate(pieces)
    ]


def candidate_segments(candidate: PatentCandidate) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    if candidate.title:
        segments.append(("title", candidate.title))
    if candidate.abstract:
        segments.append(("abstract", candidate.abstract))
    for index, claim in enumerate(candidate.claims, start=1):
        segments.append((f"claim_{index}", claim))
    return segments


def _segment_texts(candidate: PatentCandidate) -> tuple[str, ...]:
    return tuple(text for _, text in candidate_segments(candidate))


@lru_cache(maxsize=256)
def _cached_segment_bm25(corpus_texts: tuple[str, ...]) -> BM25Okapi:
    return BM25Okapi([tokenize(text) for text in corpus_texts])


def extract_evidence_for_candidate(
    limitation: ClaimLimitation,
    candidate: PatentCandidate,
) -> EvidenceMatch:
    ranked = rank_candidate_segments(limitation.text, candidate)
    source, evidence, best_score = ranked[0]
    return EvidenceMatch(
        limitation_label=limitation.label,
        limitation_text=limitation.text,
        candidate_letter=candidate.letter,
        patent_id=candidate.patent_id,
        source=source,
        evidence=evidence,
        score=float(best_score),
    )


def rank_candidate_segments(
    query_text: str,
    candidate: PatentCandidate,
) -> list[tuple[str, str, float]]:
    segments = candidate_segments(candidate)
    query = tokenize(query_text)
    bm25 = _cached_segment_bm25(_segment_texts(candidate))
    scores = bm25.get_scores(query)
    ranked = [
        (source, evidence, float(score))
        for (source, evidence), score in zip(segments, scores, strict=True)
    ]
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


def build_claim_chart(
    case: Par4pcCase,
    ranked_candidates: list[RetrievalResult],
    limitations: list[ClaimLimitation],
    top_candidates: int = 3,
) -> list[EvidenceMatch]:
    rows: list[EvidenceMatch] = []
    for result in ranked_candidates[:top_candidates]:
        candidate = case.candidates[result.letter]
        for limitation in limitations:
            rows.append(extract_evidence_for_candidate(limitation, candidate))
    return rows


def verify_evidence_heuristic(row: EvidenceMatch) -> VerificationResult:
    limitation_terms = {
        token
        for token in tokenize(row.limitation_text)
        if token not in PATENT_STOPWORDS and len(token) > 2
    }
    evidence_terms = {
        token
        for token in tokenize(row.evidence)
        if token not in PATENT_STOPWORDS and len(token) > 2
    }
    if not limitation_terms:
        return VerificationResult(status="unsupported", reason="No tokens found in limitation.")

    overlap = len(limitation_terms & evidence_terms) / len(limitation_terms)
    if overlap >= 0.50 and row.score >= 5.0:
        return VerificationResult(
            status="supported",
            reason=f"Content-token overlap={overlap:.2f}; BM25={row.score:.2f}.",
        )
    if overlap >= 0.30 and row.score >= 2.0:
        return VerificationResult(
            status="partially_supported",
            reason=f"Partial content-token overlap={overlap:.2f}; BM25={row.score:.2f}.",
        )
    return VerificationResult(
        status="unsupported",
        reason=f"Low content-token overlap={overlap:.2f}; BM25={row.score:.2f}.",
    )


def apply_verification_heuristic(chart: list[EvidenceMatch]) -> list[EvidenceMatch]:
    verified: list[EvidenceMatch] = []
    for row in chart:
        decision = verify_evidence_heuristic(row)
        verified.append(
            EvidenceMatch(
                limitation_label=row.limitation_label,
                limitation_text=row.limitation_text,
                candidate_letter=row.candidate_letter,
                patent_id=row.patent_id,
                source=row.source,
                evidence=row.evidence,
                score=row.score,
                verification=decision.status,
                verification_reason=decision.reason,
            )
        )
    return verified


def run_baseline_analysis(case: Par4pcCase, top_k: int = 3) -> tuple[list[ClaimLimitation], list[RetrievalResult], list[EvidenceMatch]]:
    limitations = decompose_claim_heuristic(case.target_claim)
    ranked = rank_candidates_bm25(case, top_k=top_k)
    chart = build_claim_chart(case, ranked, limitations, top_candidates=top_k)
    return limitations, ranked, chart


def render_report(
    case: Par4pcCase,
    limitations: list[ClaimLimitation],
    ranked: list[RetrievalResult],
    chart: list[EvidenceMatch],
) -> str:
    lines = [
        f"# Patent Prior-Art Agent MVP Report",
        "",
        f"Application: {case.application_number}",
        f"Claim: {case.claim_number}",
        f"Gold prior art: {', '.join(case.gold_answers) or 'N/A'}",
        "",
        "## Ranked Prior Art",
        "",
    ]
    for index, result in enumerate(ranked, start=1):
        lines.append(f"{index}. {result.letter} | {result.patent_id} | score={result.score:.3f} | {result.title}")

    lines.extend(["", "## Claim Limitations", ""])
    for limitation in limitations:
        lines.append(f"- {limitation.label}: {limitation.text}")

    lines.extend(["", "## Evidence Chart", ""])
    lines.append("| Limitation | Candidate | Source | BM25 | Verification | Evidence |")
    lines.append("|---|---|---|---:|---|---|")
    for row in chart:
        evidence = row.evidence.replace("|", "\\|")
        if len(evidence) > 500:
            evidence = evidence[:497] + "..."
        verification = row.verification
        if row.verification_reason:
            verification = f"{verification}: {row.verification_reason}".replace("|", "\\|")
        lines.append(
            f"| {row.limitation_label} | {row.candidate_letter} ({row.patent_id}) | "
            f"{row.source} | {row.score:.3f} | {verification} | {evidence} |"
        )
    return "\n".join(lines)
