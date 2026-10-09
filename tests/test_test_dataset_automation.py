"""
tests/test_test_dataset_automation.py

Unit and integration tests for test dataset automation and contamination prevention:
1. Canonical signature determinism & hashing
2. Primary text extraction
3. Deduplication rejection when an instance matches training data
4. Formatters validity and bounds
5. Zero-overlap verification audit
"""

import os
import json
import tempfile
import pytest

from test_dataset_automation import (
    canonical_signature,
    extract_primary_text,
    index_train_dataset,
    format_choice_with_subsampling,
    format_summeval,
    format_stsb,
    add_test_dataset,
    verify_zero_overlap,
)


def test_canonical_signature_determinism_and_insensitivity():
    rec1 = {
        "task_type": "choice",
        "domain": "finance_routing",
        "question": "What is the intent?",
        "state": {"text": "How do I check my balance?", "user_id": 123},
    }
    rec2 = {
        "task_type": "CHOICE",
        "domain": "finance_routing",
        "question": "what is the intent? ",
        "state": {"user_id": 123, "text": "how do i check my balance? "},
    }
    # Different order of keys and capitalization should yield identical signature
    assert canonical_signature(rec1) == canonical_signature(rec2)

    # Different text yields different signature
    rec3 = dict(rec1, state={"text": "How do I transfer funds?"})
    assert canonical_signature(rec1) != canonical_signature(rec3)


def test_extract_primary_text():
    rec_query = {"state": {"query": "  Send $50 to Alice.  "}}
    assert extract_primary_text(rec_query) == "send $50 to alice."

    rec_code = {"state": {"code": "def foo():\n    return 42"}}
    assert "def foo(): return 42" in extract_primary_text(rec_code)

    rec_empty = {"state": {}}
    assert extract_primary_text(rec_empty) == ""


def test_index_train_dataset():
    sample_records = [
        {"task_type": "noul", "domain": "safety", "question": "Is toxic?", "state": {"user_message": "Hello world"}},
        {"task_type": "choice", "domain": "intent", "question": "Route intent", "state": {"query": "Book flight"}},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in sample_records:
            f.write(json.dumps(r) + "\n")
        tmp_name = f.name

    try:
        sigs, texts = index_train_dataset(tmp_name)
        assert len(sigs) == 2
        assert len(texts) == 2
        assert "hello world" in texts
        assert "book flight" in texts
    finally:
        os.remove(tmp_name)


def test_add_test_dataset_rejects_train_and_duplicate_instances():
    # Simulate a training sample
    train_rec = {
        "task_type": "noul",
        "domain": "factuality",
        "question": "Is it true?",
        "state": {"passage": "Paris is the capital of France."},
        "target_bool": True,
    }
    train_sig = canonical_signature(train_rec)
    train_text = extract_primary_text(train_rec)

    train_sigs = {train_sig}
    train_texts = {train_text}

    # Candidate dataset containing:
    # 1) An exact copy of train_rec (must be rejected)
    # 2) A unique valid sample
    # 3) A duplicate of candidate 2 (must be rejected)
    # 4) Another unique valid sample
    candidates = [
        train_rec,
        {
            "task_type": "noul",
            "domain": "factuality",
            "question": "Is it true?",
            "state": {"passage": "Tokyo is the capital of Japan."},
            "target_bool": True,
        },
        {
            "task_type": "noul",
            "domain": "factuality",
            "question": "Is it true?",
            "state": {"passage": "Tokyo is the capital of Japan."},
            "target_bool": True,
        },
        {
            "task_type": "noul",
            "domain": "factuality",
            "question": "Is it true?",
            "state": {"passage": "Berlin is the capital of Germany."},
            "target_bool": True,
        },
    ]

    records = []
    test_sigs = set()
    test_texts = set()

    add_test_dataset(
        name="TestPool",
        dataset=candidates,
        formatter=lambda ex: ex,
        limit=2,
        records_list=records,
        train_signatures=train_sigs,
        train_primary_texts=train_texts,
        test_signatures=test_sigs,
        test_primary_texts=test_texts,
        seed=42,
    )

    assert len(records) == 2
    # Verify neither record is the train record
    assert not any(r["state"]["passage"] == "Paris is the capital of France." for r in records)
    # Verify the two accepted are Tokyo and Berlin
    passages = {r["state"]["passage"] for r in records}
    assert passages == {"Tokyo is the capital of Japan.", "Berlin is the capital of Germany."}


def test_format_choice_with_subsampling():
    options = ["apple", "banana", "cherry", "date", "fig"]
    result = format_choice_with_subsampling(
        state_dict={"text": "pick a fruit"},
        question="Which fruit is correct?",
        all_options=options,
        ground_truth_idx=2,  # cherry
        domain="food",
        max_distractors=3,
    )
    assert result["task_type"] == "choice"
    assert result["domain"] == "food"
    assert "cherry" in result["options"]
    assert result["options"][result["target_idx"]] == "cherry"
    assert len(result["options"]) == 4  # 1 correct + 3 distractors


def test_format_summeval_and_stsb():
    summeval_ex = {"score": 4.8, "doc": "long article", "summary": "brief summary"}
    res_summ = format_summeval(summeval_ex)
    assert res_summ["task_type"] == "score"
    assert res_summ["target_score"] == 4  # round(4.8) = 5 -> 5 - 1 = 4
    assert res_summ["active_levels"] == 5

    stsb_ex = {"score": 0.8, "sentence1": "A man plays guitar.", "sentence2": "A man makes music."}
    res_stsb = format_stsb(stsb_ex)
    assert res_stsb["task_type"] == "score"
    assert res_stsb["target_score"] == 4  # 0.8 * 5 = 4
    assert res_stsb["active_levels"] == 6


def test_verify_zero_overlap_detects_contamination():
    # Create train set with 15 domains
    domains = [
        "finance_routing", "intent_routing", "tool_routing", "factuality",
        "paraphrase", "natural_language_inference", "safety", "security",
        "content_moderation", "helpfulness_rubric", "semantic_similarity",
        "summary_evaluation", "legal", "biomedical", "code"
    ]
    train_recs = []
    test_recs = []

    for d in domains:
        train_recs.append({
            "task_type": "noul",
            "domain": d,
            "question": f"Question for {d}",
            "state": {"text": f"Train instance for domain {d}"},
            "target_bool": True,
        })
        test_recs.append({
            "task_type": "noul",
            "domain": d,
            "question": f"Question for {d}",
            "state": {"text": f"Disjoint test instance for domain {d}"},
            "target_bool": False,
        })

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f_train, \
         tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f_test:
        for r in train_recs:
            f_train.write(json.dumps(r) + "\n")
        for r in test_recs:
            f_test.write(json.dumps(r) + "\n")
        train_path = f_train.name
        test_path = f_test.name

    try:
        # Should pass clean
        audit = verify_zero_overlap(train_path, test_path)
        assert audit["passed"] is True
        assert audit["exact_hash_collisions"] == 0
        assert audit["text_collisions"] == 0

        # Inject contaminated instance into test
        with open(test_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(train_recs[0]) + "\n")

        with pytest.raises(AssertionError, match="failed contamination"):
            verify_zero_overlap(train_path, test_path)
    finally:
        if os.path.exists(train_path): os.remove(train_path)
        if os.path.exists(test_path): os.remove(test_path)
