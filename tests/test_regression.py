"""
tests/test_regression.py

Regression test suite for dataset curation formatters:
1. Dynamic permutation invariance (no positional memorization)
2. 0-indexed score bounds safety (no device-side asserts)
3. Candidate subsampling (context length control)
4. Score distribution alignment across rubrics (STS-B, SummEval, HelpSteer2)
"""

import random
import pytest
from dataset_automation import (
    format_choice_with_subsampling,
    format_summeval,
    format_stsb,
)


def test_summeval_score_bounds_and_zero_indexing():
    """Verify SummEval 1..5 scores map strictly to 0..4 range with active_levels=5."""
    for raw_score, expected_target in [(1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3), (5.0, 4)]:
        ex = {"doc": "Sample document", "summary": "Sample summary", "score": raw_score}
        record = format_summeval(ex)

        assert record["task_type"] == "score"
        assert record["active_levels"] == 5
        assert record["target_score"] == expected_target
        assert 0 <= record["target_score"] < record["active_levels"], (
            f"target_score={record['target_score']} out of bounds for active_levels={record['active_levels']}"
        )


def test_summeval_score_bounds_clamp_extremes():
    """Verify out-of-range floats still safely clamp to 0..4."""
    for raw_score, expected_target in [(0.2, 0), (5.8, 4), (-1.0, 0), (10.0, 4)]:
        ex = {"doc": "doc", "summary": "summary", "score": raw_score}
        record = format_summeval(ex)
        assert 0 <= record["target_score"] < record["active_levels"]
        assert record["target_score"] == expected_target


def test_stsb_score_bounds_and_scaling():
    """Verify STS-B 0..1 float scores map to 0..5 integer rubric with active_levels=6."""
    test_cases = [
        (0.0, 0),
        (0.2, 1),
        (0.4, 2),
        (0.6, 3),
        (0.8, 4),
        (1.0, 5),
    ]
    for sim, expected in test_cases:
        ex = {"sentence1": "s1", "sentence2": "s2", "score": sim}
        record = format_stsb(ex)
        assert record["task_type"] == "score"
        assert record["active_levels"] == 6
        assert record["target_score"] == expected
        assert 0 <= record["target_score"] < record["active_levels"]


def test_choice_subsampling_high_cardinality():
    """Verify Banking77 (77 labels) and CLINC150 (151 labels) are capped at max_distractors + 1."""
    rng = random.Random(42)
    all_options = [f"intent_{i}" for i in range(151)]
    ground_truth_idx = 73
    correct_label = all_options[ground_truth_idx]

    record = format_choice_with_subsampling(
        state_dict={"query": "test query"},
        question="Select intent",
        all_options=all_options,
        ground_truth_idx=ground_truth_idx,
        domain="intent_routing",
        max_distractors=9,
        rng=rng,
    )

    assert len(record["options"]) == 10  # 1 correct + 9 distractors
    assert record["options"][record["target_idx"]] == correct_label
    assert 0 <= record["target_idx"] < len(record["options"])


def test_choice_low_cardinality_retains_all_options_and_permutes():
    """Verify low cardinality datasets (ANLI, PubMedQA) keep all options and dynamically shuffle."""
    rng = random.Random(42)
    options = ["entailment", "neutral", "contradiction"]
    ground_truth_idx = 0  # "entailment"

    permutations_seen = set()
    target_indices_seen = set()

    for _ in range(50):
        record = format_choice_with_subsampling(
            state_dict={"premise": "p", "hypothesis": "h"},
            question="Relationship",
            all_options=options,
            ground_truth_idx=ground_truth_idx,
            domain="natural_language_inference",
            max_distractors=9,
            rng=rng,
        )

        assert len(record["options"]) == 3
        assert set(record["options"]) == set(options)
        assert record["options"][record["target_idx"]] == "entailment"
        permutations_seen.add(tuple(record["options"]))
        target_indices_seen.add(record["target_idx"])

    # Must see multiple permutations and dynamic target positions (not locked to index 0)
    assert len(permutations_seen) > 1, "Options were never permuted!"
    assert len(target_indices_seen) > 1, "Target index was always fixed to the same slot!"


def test_choice_pointer_invariant_across_1000_fuzz_trials():
    """Fuzz test 1,000 arbitrary choice instances to guarantee target_idx pointer integrity."""
    rng = random.Random(1337)
    for trial in range(1000):
        num_candidates = rng.randint(2, 200)
        max_distractors = rng.randint(2, 15)
        all_options = [f"opt_{trial}_{i}" for i in range(num_candidates)]
        gt_idx = rng.randint(0, num_candidates - 1)
        expected_label = all_options[gt_idx]

        record = format_choice_with_subsampling(
            state_dict={"t": trial},
            question="Fuzz query",
            all_options=all_options,
            ground_truth_idx=gt_idx,
            domain="fuzz_domain",
            max_distractors=max_distractors,
            rng=rng,
        )

        # Options size check
        expected_len = min(num_candidates, max_distractors + 1)
        assert len(record["options"]) == expected_len
        # Target pointer check
        assert record["options"][record["target_idx"]] == expected_label
        assert 0 <= record["target_idx"] < len(record["options"])
