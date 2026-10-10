"""
tests/test_dataset_bugs.py

Reproduction and regression harness for the three critical dataset bugs:
1. Static Option Ordering in `choice` (Positional Memorization)
2. Score Indexing Discrepancy & Bounds Assertion Crash
3. Candidate Option Inflation on Context Length
"""

import json
import os
import pytest
from collections import defaultdict

DATASET_PATH = "data/decision_train_26k.jsonl" if os.path.exists("data/decision_train_26k.jsonl") else "decision_train_26k.jsonl"


@pytest.fixture(scope="module")
def dataset_records():
    assert os.path.exists(DATASET_PATH), f"Dataset file {DATASET_PATH} not found!"
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def test_score_target_within_bounds(dataset_records):
    """
    Critical Issue 2: Score Indexing Discrepancy & Bounds Assertion Crash
    Every target_score must be 0-indexed: 0 <= target_score < active_levels.
    When active_levels is 5, target_score must be in {0, 1, 2, 3, 4}.
    """
    score_records = [r for r in dataset_records if r.get("task_type") == "score"]
    assert len(score_records) > 0, "No score records found in dataset"

    out_of_bounds = []
    for idx, r in enumerate(score_records):
        target = r.get("target_score")
        active_levels = r.get("active_levels")
        domain = r.get("domain")

        # PyTorch slicing assertion check: target must be valid index for dimension with size active_levels
        if not (0 <= target < active_levels):
            out_of_bounds.append({
                "sample_index": idx,
                "domain": domain,
                "target_score": target,
                "active_levels": active_levels,
            })

    assert len(out_of_bounds) == 0, (
        f"Found {len(out_of_bounds)} score samples out of bounds (target_score >= active_levels). "
        f"First few failures: {out_of_bounds[:5]}"
    )


def test_choice_permutation_invariance(dataset_records):
    """
    Critical Issue 1: Static Option Ordering in `choice` (Positional Memorization)
    Choice options must be dynamically shuffled across instances to prevent
    positional slot memorization. No domain should have identical option ordering across all samples.
    """
    choice_records = defaultdict(list)
    for r in dataset_records:
        if r.get("task_type") == "choice":
            choice_records[r.get("domain")].append(r)

    assert len(choice_records) > 0, "No choice records found in dataset"

    static_order_domains = []
    for domain, recs in choice_records.items():
        if len(recs) <= 1:
            continue
        first_options = tuple(recs[0]["options"])
        all_identical = all(tuple(r["options"]) == first_options for r in recs)
        if all_identical:
            static_order_domains.append((domain, len(recs)))

    assert len(static_order_domains) == 0, (
        f"Domains with static option ordering found: {static_order_domains}. "
        f"Options must be dynamically permuted per sample."
    )


def test_candidate_option_subsampling(dataset_records):
    """
    Critical Issue 3: Candidate Option Inflation on Context Length
    For intent routing with high label counts (e.g. Banking77 with 77 labels, CLINC150 with 151 labels),
    candidates should be subsampled (ground truth + max 9 distractors <= 10 candidates).
    """
    high_cardinality_domains = ["finance_routing", "intent_routing"]
    bloated_samples = []

    for idx, r in enumerate(dataset_records):
        domain = r.get("domain")
        if domain in high_cardinality_domains:
            num_options = len(r.get("options", []))
            if num_options > 10:
                bloated_samples.append({
                    "sample_index": idx,
                    "domain": domain,
                    "num_options": num_options,
                })

    assert len(bloated_samples) == 0, (
        f"Found {len(bloated_samples)} samples in high-cardinality routing domains with > 10 options. "
        f"First few bloated samples: {bloated_samples[:5]}"
    )
