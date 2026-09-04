from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PatentCandidate:
    letter: str
    patent_id: str
    title: str
    abstract: str
    claims: list[str]

    @property
    def retrieval_text(self) -> str:
        return "\n".join(
            part
            for part in [
                self.title,
                self.abstract,
                "\n".join(self.claims),
            ]
            if part
        )


@dataclass(frozen=True)
class Par4pcCase:
    source_path: Path
    application_number: str
    claim_number: int
    title: str
    abstract: str
    claims: list[str]
    target_claim: str
    candidates: dict[str, PatentCandidate]
    gold_answers: list[str]
    silver_answers: list[str]
    negative_answers: list[str]


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_letters(value: Any) -> list[str]:
    return sorted({str(item).strip().upper() for item in _as_str_list(value) if str(item).strip()})


def load_par4pc_case(path: str | Path) -> Par4pcCase:
    source_path = Path(path)
    data = json.loads(source_path.read_text(encoding="utf-8"))

    context = data.get("context") or {}
    claims = _as_str_list(context.get("claims"))
    claim_number = int(data["claim_number"])
    target_index = claim_number - 1
    if target_index < 0 or target_index >= len(claims):
        raise ValueError(f"Claim {claim_number} not found in {source_path}")

    candidates: dict[str, PatentCandidate] = {}
    for letter, details in sorted((data.get("options") or {}).items()):
        candidates[str(letter).upper()] = PatentCandidate(
            letter=str(letter).upper(),
            patent_id=str(details.get("patent_id", "")),
            title=str(details.get("title", "")),
            abstract=str(details.get("abstract", "")),
            claims=_as_str_list(details.get("claims")),
        )

    return Par4pcCase(
        source_path=source_path,
        application_number=str(data.get("application_number", "")),
        claim_number=claim_number,
        title=str(context.get("title", "")),
        abstract=str(context.get("abstract", "")),
        claims=claims,
        target_claim=claims[target_index],
        candidates=candidates,
        gold_answers=_normalize_letters(data.get("gold_answers")),
        silver_answers=_normalize_letters(data.get("silver_answers")),
        negative_answers=_normalize_letters(data.get("negative_answers")),
    )


def load_par4pc_dir(data_dir: str | Path) -> list[Par4pcCase]:
    root = Path(data_dir)
    paths = sorted(root.glob("par4pc_*.json"))
    if not paths:
        raise FileNotFoundError(f"No PAR4PC JSON files found in {root}")
    return [load_par4pc_case(path) for path in paths]


def load_unique_patent_pool(data_dir: str | Path) -> list[PatentCandidate]:
    cases = load_par4pc_dir(data_dir)
    patents: dict[str, PatentCandidate] = {}
    for case in cases:
        for candidate in case.candidates.values():
            patents.setdefault(candidate.patent_id or candidate.letter, candidate)
    return sorted(patents.values(), key=lambda item: (item.patent_id, item.title))


def _candidate_from_details(letter: str, details: Any) -> PatentCandidate:
    details = details or {}
    return PatentCandidate(
        letter=str(letter).upper(),
        patent_id=str(details.get("patent_id", "")),
        title=str(details.get("title", "")),
        abstract=str(details.get("abstract", "")),
        claims=_as_str_list(details.get("claims")),
    )


def load_hf_par4pc_patent_pool(
    splits: tuple[str, ...] = ("train", "validation", "test"),
    max_rows_per_split: int | None = 2000,
    repo_id: str = "LG-AI-Research/PANORAMA",
) -> list[PatentCandidate]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    patents: dict[str, PatentCandidate] = {}
    for split in splits:
        parquet_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"PAR4PC/{split}.parquet",
        )
        parquet_file = pq.ParquetFile(parquet_path)
        rows_loaded = 0
        stop = False
        for batch in parquet_file.iter_batches(columns=["options"], batch_size=256):
            for row in batch.to_pylist():
                options = row.get("options") or {}
                for letter, details in sorted(options.items()):
                    candidate = _candidate_from_details(letter, details)
                    patents.setdefault(candidate.patent_id or candidate.letter, candidate)
                rows_loaded += 1
                if max_rows_per_split is not None and rows_loaded >= max_rows_per_split:
                    stop = True
                    break
            if stop:
                break
    return sorted(patents.values(), key=lambda item: (item.patent_id, item.title))


def load_hf_par4pc_cases(
    splits: tuple[str, ...] = ("validation",),
    max_rows_per_split: int | None = 200,
    repo_id: str = "LG-AI-Research/PANORAMA",
) -> list[Par4pcCase]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    cases: list[Par4pcCase] = []
    for split in splits:
        parquet_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"PAR4PC/{split}.parquet",
        )
        parquet_file = pq.ParquetFile(parquet_path)
        rows_loaded = 0
        stop = False
        for batch in parquet_file.iter_batches(
            columns=[
                "application_number",
                "claim_number",
                "context",
                "options",
                "gold_answers",
                "silver_answers",
                "negative_answers",
            ],
            batch_size=256,
        ):
            for row in batch.to_pylist():
                context = row.get("context") or {}
                claims = _as_str_list(context.get("claims"))
                claim_number = int(row["claim_number"])
                target_index = claim_number - 1
                if target_index < 0 or target_index >= len(claims):
                    continue

                candidates: dict[str, PatentCandidate] = {}
                for letter, details in sorted((row.get("options") or {}).items()):
                    candidates[str(letter).upper()] = _candidate_from_details(letter, details)

                cases.append(
                    Par4pcCase(
                        source_path=Path(f"hf://{repo_id}/PAR4PC/{split}#{rows_loaded}"),
                        application_number=str(row.get("application_number", "")),
                        claim_number=claim_number,
                        title=str(context.get("title", "")),
                        abstract=str(context.get("abstract", "")),
                        claims=claims,
                        target_claim=claims[target_index],
                        candidates=candidates,
                        gold_answers=_normalize_letters(row.get("gold_answers")),
                        silver_answers=_normalize_letters(row.get("silver_answers")),
                        negative_answers=_normalize_letters(row.get("negative_answers")),
                    )
                )
                rows_loaded += 1
                if max_rows_per_split is not None and rows_loaded >= max_rows_per_split:
                    stop = True
                    break
            if stop:
                break
    return cases


def combine_patent_pools(*pools: list[PatentCandidate]) -> list[PatentCandidate]:
    patents: dict[str, PatentCandidate] = {}
    for pool in pools:
        for candidate in pool:
            patents.setdefault(candidate.patent_id or candidate.letter, candidate)
    return sorted(patents.values(), key=lambda item: (item.patent_id, item.title))
