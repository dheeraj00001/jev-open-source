"""
test_dataset_automation.py

Automated downloading, curating, and contamination-free verification of a 2,600-sample
test dataset for the System One Hybrid Decision Model.

Guarantees 100% disjointness from training and validation data (decision_train_26k.jsonl)
using multi-layer signature and primary text indexing across all 15 domains.

Usage:
    # 1. Download, curate, and verify test dataset:
    python test_dataset_automation.py --train_path decision_train_26k.jsonl --output_file decision_test_2.6k.jsonl

    # 2. Evaluate trained model on test dataset:
    python test_dataset_automation.py --evaluate_model_dir ./system_one_model --output_file decision_test_2.6k.jsonl
"""

import os
import json
import random
import hashlib
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional
from datasets import load_dataset

SEED_TEST = 1337


# ==============================================================================
# 1. SIGNATURE & CONTAMINATION DETECTION ENGINE
# ==============================================================================

def canonical_signature(record: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA256 digest over task_type, domain, question,
    and all normalized state fields.
    """
    task = record.get("task_type", "").strip().lower()
    domain = record.get("domain", "").strip().lower()
    question = record.get("question", "").strip().lower()

    state = record.get("state", {})
    state_items = []
    for k in sorted(state.keys()):
        v = state[k]
        if isinstance(v, str):
            state_items.append(f"{k}:{v.strip().lower()}")
        elif isinstance(v, (int, float, bool)):
            state_items.append(f"{k}:{v}")
        elif isinstance(v, list):
            state_items.append(f"{k}:{json.dumps(v, sort_keys=True)}")
        elif isinstance(v, dict):
            state_items.append(f"{k}:{json.dumps(v, sort_keys=True)}")

    payload = f"{task}|{domain}|{question}|" + "|".join(state_items)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_primary_text(record: Dict[str, Any]) -> str:
    """
    Extracts and normalizes the core textual payload from the state dictionary.
    For pairwise tasks (document/summary, premise/hypothesis, sentence1/sentence2, prompt/response),
    combines both fields so that the complete decision context is represented.
    """
    state = record.get("state", {})
    if not isinstance(state, dict):
        return ""

    # Specific multi-part state handling
    if ("document" in state or "doc" in state) and "summary" in state:
        doc = state.get("document", state.get("doc", ""))
        summary = state.get("summary", "")
        return f"{doc.strip().lower()} ||| {summary.strip().lower()}"

    if "sentence1" in state and "sentence2" in state:
        return f"{state['sentence1'].strip().lower()} ||| {state['sentence2'].strip().lower()}"

    if "premise" in state and "hypothesis" in state:
        return f"{state['premise'].strip().lower()} ||| {state['hypothesis'].strip().lower()}"

    if "prompt" in state and "response" in state:
        return f"{state['prompt'].strip().lower()} ||| {state['response'].strip().lower()}"

    # Single-text primary fields
    for key in [
        "text", "query", "passage", "prompt", "user_message",
        "code", "context", "document", "doc"
    ]:
        if key in state and isinstance(state[key], str) and state[key].strip():
            normalized = " ".join(state[key].strip().lower().split())
            if len(normalized) > 10:
                return normalized

    # Fallback to serialized sorted state values
    return " ".join(str(v).strip().lower() for v in state.values() if isinstance(v, str))


def index_train_dataset(train_path: str) -> Tuple[Set[str], Set[str]]:
    """
    Indexes all canonical signatures and primary texts from the training dataset.
    """
    print(f"\n[Deduplication Indexer] Loading and indexing training data from: {train_path}...")
    train_signatures: Set[str] = set()
    train_primary_texts: Set[str] = set()

    if not os.path.exists(train_path):
        print(f"  Warning: Training file {train_path} not found. Proceeding with empty exclusion index.")
        return train_signatures, train_primary_texts

    count = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sig = canonical_signature(rec)
            train_signatures.add(sig)

            prim_text = extract_primary_text(rec)
            if prim_text:
                train_primary_texts.add(prim_text)
            count += 1

    print(f"  Indexed {count:,} training samples -> {len(train_signatures):,} canonical signatures, {len(train_primary_texts):,} primary texts.")
    return train_signatures, train_primary_texts


# ==============================================================================
# 2. FORMATTERS
# ==============================================================================

def format_choice_with_subsampling(
    state_dict: dict,
    question: str,
    all_options: list,
    ground_truth_idx: int,
    domain: str,
    max_distractors: int = 9,
    rng: Optional[random.Random] = None,
) -> dict:
    if rng is None:
        rng = random

    correct_option = all_options[ground_truth_idx]
    distractors = [opt for i, opt in enumerate(all_options) if i != ground_truth_idx]

    if len(distractors) > max_distractors:
        selected_distractors = rng.sample(distractors, max_distractors)
    else:
        selected_distractors = list(distractors)

    indexed_candidates = [(True, correct_option)] + [(False, opt) for opt in selected_distractors]
    rng.shuffle(indexed_candidates)
    candidates = [opt for _, opt in indexed_candidates]
    target_idx = [i for i, (is_correct, _) in enumerate(indexed_candidates) if is_correct][0]

    return {
        "task_type": "choice",
        "domain": domain,
        "state": state_dict,
        "question": question,
        "options": candidates,
        "target_idx": target_idx,
    }


def format_summeval(ex: dict) -> dict:
    raw_score = round(ex["score"])
    normalized_target = max(0, min(int(raw_score) - 1, 4))
    return {
        "task_type": "score",
        "domain": "summary_evaluation",
        "state": {
            "document": ex.get("text", ex.get("document", ex.get("doc", ""))),
            "summary": ex.get("summary", ""),
        },
        "question": "Rate the quality of this summary on the evaluation rubric.",
        "target_score": normalized_target,
        "active_levels": 5,
    }


def format_stsb(ex: dict) -> dict:
    raw_val = ex["score"]
    scaled = raw_val * 5.0 if raw_val <= 1.0 else raw_val
    clamped = min(max(scaled, 0.0), 5.0)
    return {
        "task_type": "score",
        "domain": "semantic_similarity",
        "state": {
            "sentence1": ex["sentence1"],
            "sentence2": ex["sentence2"],
        },
        "question": "Rate the semantic similarity of sentence1 and sentence2 from 0 to 5.",
        "target_score": int(round(clamped)),
        "active_levels": 6,
    }


# ==============================================================================
# 3. DISJOINT SAMPLE INGESTION
# ==============================================================================

def add_test_dataset(
    name: str,
    dataset: Any,
    formatter: Any,
    limit: int,
    records_list: List[dict],
    train_signatures: Set[str],
    train_primary_texts: Set[str],
    test_signatures: Set[str],
    test_primary_texts: Set[str],
    seed: int = SEED_TEST,
):
    """
    Shuffles dataset pool and selects samples with strict verification
    that zero instances overlap with training data or existing test samples.
    """
    print(f"\n[{name}]")
    total_avail = len(dataset)
    print(f"Available candidates: {total_avail:,} | Requested: {limit:,}")

    if hasattr(dataset, "shuffle"):
        shuffled = dataset.shuffle(seed=seed)
    else:
        rng = random.Random(seed)
        shuffled = list(dataset)
        rng.shuffle(shuffled)

    added = 0
    rejected_train = 0
    rejected_dup = 0

    for ex in shuffled:
        try:
            rec = formatter(ex)
        except Exception:
            continue

        sig = canonical_signature(rec)
        prim_text = extract_primary_text(rec)

        # Check 1: Must NOT exist in training/validation data
        if sig in train_signatures:
            rejected_train += 1
            continue
        if prim_text and prim_text in train_primary_texts:
            rejected_train += 1
            continue

        # Check 2: Must NOT be duplicate within test set
        if sig in test_signatures:
            rejected_dup += 1
            continue
        if prim_text and prim_text in test_primary_texts:
            rejected_dup += 1
            continue

        # Valid disjoint sample
        records_list.append(rec)
        test_signatures.add(sig)
        if prim_text:
            test_primary_texts.add(prim_text)
        added += 1

        if added >= limit:
            break

    print(f"Added: {added:,} (Rejected {rejected_train} train-overlap, {rejected_dup} internal duplicates)")

    if added < limit:
        raise ValueError(
            f"Dataset {name}: could only collect {added} unique non-overlapping samples, "
            f"but {limit} were requested."
        )


# ==============================================================================
# 4. BUILD COMPLETE 2,600-SAMPLE TEST DATASET
# ==============================================================================

def build_test_dataset(
    train_path: str = "decision_train_26k.jsonl",
    output_file: str = "decision_test_2.6k.jsonl",
    seed: int = SEED_TEST,
) -> List[dict]:
    random.seed(seed)
    rng = random.Random(seed)

    # 1. Build training exclusion indices
    train_signatures, train_primary_texts = index_train_dataset(train_path)

    records: List[dict] = []
    test_signatures: Set[str] = set()
    test_primary_texts: Set[str] = set()

    # ============================================================
    # 1. AGENTIC ROUTING & INTENT (Target: 650)
    # ============================================================
    print("\n" + "=" * 60)
    print("1. AGENTIC ROUTING & INTENT (Test Split)")
    print("=" * 60)

    # Banking77 — 300 from official test split
    banking_test = load_dataset("mteb/banking77", split="test")
    banking_labels = [banking_test.features["label_text"].names[i] if hasattr(banking_test.features["label_text"], "names")
                      else banking_test[i]["label_text"] for i in range(77)] if hasattr(banking_test, "features") and "label_text" in banking_test.features else None
    if not banking_labels:
        # Build map
        lmap = {}
        for ex in banking_test:
            lmap[ex["label"]] = ex["label_text"]
            if len(lmap) == 77:
                break
        banking_labels = [lmap[i] for i in range(len(lmap))]

    add_test_dataset(
        "Banking77 (test split)",
        banking_test,
        lambda ex: format_choice_with_subsampling(
            state_dict={"text": ex["text"]},
            question="Identify the primary customer service intent for this banking query.",
            all_options=banking_labels,
            ground_truth_idx=ex["label"],
            domain="finance_routing",
            max_distractors=9,
            rng=rng,
        ),
        limit=300,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # CLINC150 — 250 from official test split
    clinc_test = load_dataset("clinc/clinc_oos", "plus", split="test")
    clinc_labels = clinc_test.features["intent"].names

    add_test_dataset(
        "CLINC150 (test split)",
        clinc_test,
        lambda ex: format_choice_with_subsampling(
            state_dict={"query": ex["text"]},
            question="Route this utterance to the correct application intent or out-of-scope.",
            all_options=clinc_labels,
            ground_truth_idx=ex["intent"],
            domain="intent_routing",
            max_distractors=9,
            rng=rng,
        ),
        limit=250,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # xLAM / Apigen Tool Routing — 100 from unseen instances
    apigen_raw = load_dataset("argilla/apigen-function-calling", split="train")

    def is_valid_tool_call(ex):
        try:
            if not ex.get("tools") or not ex.get("answers") or not ex.get("query"):
                return False
            tools = json.loads(ex["tools"])
            answers = json.loads(ex["answers"])
            if not isinstance(tools, list) or not isinstance(answers, list) or len(tools) == 0 or len(answers) == 0:
                return False
            tool_names = [t.get("name", "") for t in tools if isinstance(t, dict)]
            called_name = answers[0].get("name", "") if isinstance(answers[0], dict) else ""
            return bool(called_name and called_name in tool_names and len(tool_names) >= 2)
        except Exception:
            return False

    tool_routing_test = apigen_raw.filter(is_valid_tool_call)

    def format_tool_routing(ex):
        tools = json.loads(ex["tools"])
        tool_names = [t["name"] for t in tools]
        answers = json.loads(ex["answers"])
        called_name = answers[0]["name"]
        return format_choice_with_subsampling(
            state_dict={"query": ex["query"]},
            question="Select the appropriate tool or function for this user request.",
            all_options=tool_names,
            ground_truth_idx=tool_names.index(called_name),
            domain="tool_routing",
            max_distractors=9,
            rng=rng,
        )

    add_test_dataset(
        "xLAM / Apigen Tool Routing (unseen instances)",
        tool_routing_test,
        format_tool_routing,
        limit=100,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # ============================================================
    # 2. FACTUALITY & GROUNDING (Target: 650)
    # ============================================================
    print("\n" + "=" * 60)
    print("2. FACTUALITY & GROUNDING (Validation & Test Splits)")
    print("=" * 60)

    # BoolQ — 250 from validation split
    boolq_val = load_dataset("google/boolq", split="validation")
    add_test_dataset(
        "BoolQ (validation split)",
        boolq_val,
        lambda ex: {
            "task_type": "noul",
            "domain": "factuality",
            "state": {"passage": ex["passage"]},
            "question": f"Based on the passage, is the answer to '{ex['question']}' yes?",
            "target_bool": bool(ex["answer"]),
        },
        limit=250,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # PAWS — 200 from official test split
    paws_test = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
    add_test_dataset(
        "PAWS (test split)",
        paws_test,
        lambda ex: {
            "task_type": "noul",
            "domain": "paraphrase",
            "state": {"sentence1": ex["sentence1"], "sentence2": ex["sentence2"]},
            "question": "Are sentence1 and sentence2 semantically equivalent paraphrases?",
            "target_bool": bool(ex["label"] == 1),
        },
        limit=200,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # ANLI — 200 from official test_r1 split
    anli_test = load_dataset("facebook/anli", split="test_r1")
    anli_options = ["entailment", "neutral", "contradiction"]
    add_test_dataset(
        "ANLI (test_r1 split)",
        anli_test,
        lambda ex: format_choice_with_subsampling(
            state_dict={"premise": ex["premise"], "hypothesis": ex["hypothesis"]},
            question="Determine the relationship between the premise and hypothesis.",
            all_options=anli_options,
            ground_truth_idx=ex["label"],
            domain="natural_language_inference",
            max_distractors=9,
            rng=rng,
        ),
        limit=200,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # ============================================================
    # 3. SAFETY & GUARDRAILS (Target: 500)
    # ============================================================
    print("\n" + "=" * 60)
    print("3. SAFETY & GUARDRAILS (Test Splits & Unseen Instances)")
    print("=" * 60)

    # ToxicChat — 300 from official test split
    toxic_test = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    add_test_dataset(
        "ToxicChat (test split)",
        toxic_test,
        lambda ex: {
            "task_type": "noul",
            "domain": "safety",
            "state": {"user_message": ex["user_input"]},
            "question": "Does this user message contain toxic, abusive, or harmful content?",
            "target_bool": bool(ex["toxicity"] == 1),
        },
        limit=300,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # Prompt Injections — 80 from official test split
    prompt_inj_test = load_dataset("deepset/prompt-injections", split="test")
    add_test_dataset(
        "Prompt Injections (test split)",
        prompt_inj_test,
        lambda ex: {
            "task_type": "noul",
            "domain": "security",
            "state": {"prompt": ex["text"]},
            "question": "Is this input a prompt injection or jailbreak attempt?",
            "target_bool": bool(ex["label"] == 1),
        },
        limit=80,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # OpenAI Moderation — 120 from remaining unseen instances
    mod_raw = load_dataset("mmathys/openai-moderation-api-evaluation", split="train")
    MOD_POLICY_COLS = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]
    add_test_dataset(
        "OpenAI Moderation (unseen instances)",
        mod_raw,
        lambda ex: {
            "task_type": "noul",
            "domain": "content_moderation",
            "state": {"text": ex["prompt"]},
            "question": "Does this content violate any applicable safety policy?",
            "target_bool": bool(any(ex.get(k, 0) == 1 for k in MOD_POLICY_COLS)),
        },
        limit=120,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # ============================================================
    # 4. RUBRIC SCORING & EVALUATION (Target: 500)
    # ============================================================
    print("\n" + "=" * 60)
    print("4. RUBRIC SCORING & EVALUATION (Test & Validation Splits)")
    print("=" * 60)

    # HelpSteer2 — 300 from validation split
    helpsteer_val = load_dataset("nvidia/HelpSteer2", split="validation")
    add_test_dataset(
        "HelpSteer2 (validation split)",
        helpsteer_val,
        lambda ex: {
            "task_type": "score",
            "domain": "helpfulness_rubric",
            "state": {"prompt": ex["prompt"], "response": ex["response"]},
            "question": "Rate the overall helpfulness of the response on a 0 to 4 rubric scale.",
            "target_score": int(ex["helpfulness"]),
            "active_levels": 5,
        },
        limit=300,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # STS-B — 120 from official test split
    stsb_test = load_dataset("sentence-transformers/stsb", split="test")
    add_test_dataset(
        "STS-B (test split)",
        stsb_test,
        format_stsb,
        limit=120,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # SummEval — 80 from unseen pairs in test split
    summeval_raw = load_dataset("mteb/summeval", split="test")
    summeval_pairs = []
    for ex in summeval_raw:
        doc = ex.get("text", "")
        for summary, score in zip(ex.get("machine_summaries", []), ex.get("relevance", [])):
            summeval_pairs.append({"doc": doc, "summary": summary, "score": score})

    add_test_dataset(
        "SummEval (unseen evaluation pairs)",
        summeval_pairs,
        format_summeval,
        limit=80,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # ============================================================
    # 5. SPECIALIZED VERTICALS (Target: 300)
    # ============================================================
    print("\n" + "=" * 60)
    print("5. SPECIALIZED VERTICALS (Test Splits & Artificial Pool)")
    print("=" * 60)

    # LexGLUE — 120 from official test split
    lexglue_test = load_dataset("coastalcph/lex_glue", "unfair_tos", split="test")
    add_test_dataset(
        "LexGLUE (test split)",
        lexglue_test,
        lambda ex: {
            "task_type": "noul",
            "domain": "legal",
            "state": {"text": ex["text"]},
            "question": "Does this legal clause contain an unfair or potentially unlawful provision?",
            "target_bool": bool(len(ex["labels"]) > 0),
        },
        limit=120,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # PubMedQA — 100 from pqa_artificial unseen split
    pubmed_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    med_options = ["yes", "no", "maybe"]
    med_map = {"yes": 0, "no": 1, "maybe": 2}

    add_test_dataset(
        "PubMedQA (pqa_artificial pool)",
        pubmed_art,
        lambda ex: format_choice_with_subsampling(
            state_dict={"context": " ".join(ex["context"]["contexts"]) if isinstance(ex["context"], dict) else str(ex["context"])},
            question=ex["question"],
            all_options=med_options,
            ground_truth_idx=med_map.get(ex["final_decision"], 2),
            domain="biomedical",
            max_distractors=9,
            rng=rng,
        ),
        limit=100,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # Code Defect Detection — 80 from official test split
    code_test = load_dataset("google/code_x_glue_cc_defect_detection", split="test")
    add_test_dataset(
        "Code Defect Detection (test split)",
        code_test,
        lambda ex: {
            "task_type": "noul",
            "domain": "code",
            "state": {"code": ex.get("func", "")},
            "question": "Does this code contain a software defect or vulnerability?",
            "target_bool": bool(ex["target"]),
        },
        limit=80,
        records_list=records,
        train_signatures=train_signatures,
        train_primary_texts=train_primary_texts,
        test_signatures=test_signatures,
        test_primary_texts=test_primary_texts,
        seed=seed,
    )

    # ============================================================
    # FINAL SHUFFLE & EXPORT
    # ============================================================
    print("\n" + "=" * 60)
    print("FINALIZING TEST DATASET")
    print("=" * 60)

    random.shuffle(records)
    total_samples = len(records)
    expected_samples = 2600

    print(f"Total curated test samples: {total_samples:,} (Expected: {expected_samples:,})")
    assert total_samples == expected_samples, f"Sample count mismatch: {total_samples} != {expected_samples}"

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Successfully saved test dataset to: {output_file}")

    # Run verification immediately
    verify_zero_overlap(train_path=train_path, test_path=output_file)

    return records


# ==============================================================================
# 5. ZERO-OVERLAP & INTEGRITY VERIFICATION AUDIT
# ==============================================================================

def verify_zero_overlap(train_path: str, test_path: str) -> Dict[str, Any]:
    """
    Rigorously evaluates the test dataset against the training dataset:
    - 0 canonical signature collisions
    - 0 primary text collisions
    - 100% schema compliance across all task types
    """
    print("\n" + "=" * 60)
    print("EXECUTING ZERO-OVERLAP & CONTAMINATION VERIFICATION AUDIT")
    print("=" * 60)

    train_signatures, train_primary_texts = index_train_dataset(train_path)

    test_count = 0
    test_signatures: Set[str] = set()
    test_primary_texts: Set[str] = set()

    exact_hash_collisions = 0
    text_collisions = 0
    schema_errors = 0
    domain_counts: Dict[str, int] = {}
    task_counts: Dict[str, int] = {}

    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            test_count += 1
            rec = json.loads(line)

            # Schema validation
            task = rec.get("task_type")
            domain = rec.get("domain")
            state = rec.get("state")
            question = rec.get("question")

            if not task or not domain or not isinstance(state, dict) or not question:
                schema_errors += 1
                continue

            if task == "choice":
                opts = rec.get("options")
                t_idx = rec.get("target_idx")
                if not isinstance(opts, list) or len(opts) < 2 or not isinstance(t_idx, int) or t_idx < 0 or t_idx >= len(opts):
                    schema_errors += 1
            elif task == "noul":
                if "target_bool" not in rec or not isinstance(rec["target_bool"], bool):
                    schema_errors += 1
            elif task == "score":
                sc = rec.get("target_score")
                al = rec.get("active_levels")
                if not isinstance(sc, int) or not isinstance(al, int) or sc < 0 or sc >= al:
                    schema_errors += 1
            else:
                schema_errors += 1

            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            task_counts[task] = task_counts.get(task, 0) + 1

            # Contamination checks
            sig = canonical_signature(rec)
            if sig in train_signatures:
                exact_hash_collisions += 1

            prim = extract_primary_text(rec)
            if prim and prim in train_primary_texts:
                text_collisions += 1

            test_signatures.add(sig)
            if prim:
                test_primary_texts.add(prim)

    print(f"\n[Audit Summary]")
    print(f"  Test samples verified    : {test_count:,}")
    print(f"  Unique test signatures   : {len(test_signatures):,}")
    print(f"  Schema errors            : {schema_errors}")
    print(f"  Exact Hash Collisions    : {exact_hash_collisions} (Expected: 0)")
    print(f"  Primary Text Collisions  : {text_collisions} (Expected: 0)")

    print("\nTask Distribution:")
    for t, c in sorted(task_counts.items()):
        print(f"  {t:10s}: {c:,} ({c/test_count*100:.1f}%)")

    print(f"\nDomain Coverage ({len(domain_counts)}/15 domains):")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d:30s}: {c:,}")

    passed = (exact_hash_collisions == 0 and text_collisions == 0 and schema_errors == 0 and len(domain_counts) == 15)

    if passed:
        print("\n" + "=" * 60)
        print("VERIFICATION RESULT: PASSED (ZERO CONTAMINATION CONFIRMED)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("VERIFICATION RESULT: FAILED")
        print("=" * 60)
        raise AssertionError("Test dataset failed contamination or schema audit!")

    return {
        "test_count": test_count,
        "exact_hash_collisions": exact_hash_collisions,
        "text_collisions": text_collisions,
        "schema_errors": schema_errors,
        "domains": domain_counts,
        "tasks": task_counts,
        "passed": passed,
    }


# ==============================================================================
# 6. MODEL EVALUATION ON UNSEEN TEST DATASET
# ==============================================================================

def evaluate_test_dataset_with_model(
    model_dir: str = "./system_one_model",
    test_path: str = "decision_test_2.6k.jsonl",
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluates the trained System One model directly on the unseen test dataset.
    """
    import torch
    from torch.utils.data import DataLoader
    from fine_tuning_engine import (
        DecisionDataset,
        DecisionDataCollator,
        HybridDecisionEngine,
        DecisionLossEngine,
        evaluate_model,
    )
    from safetensors.torch import load_file

    print("\n" + "=" * 60)
    print(f"EVALUATING MODEL ON HELD-OUT TEST DATASET: {test_path}")
    print(f"Model Directory: {model_dir}")
    print("=" * 60)

    if not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"

    dataset = DecisionDataset(test_path)
    loss_engine = DecisionLossEngine().to(device)

    model = HybridDecisionEngine(
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        hidden_dim=1536,
        proj_dim=512,
        load_in_4bit=(device.startswith("cuda")),
        target_device=device,
    )

    # Load LoRA adapters
    lora_dir = os.path.join(model_dir, "lora_adapters")
    if os.path.exists(lora_dir):
        print(f"Loading LoRA adapters from {lora_dir}...")
        model.backbone.load_adapter(lora_dir, adapter_name="default")

    # Load Hybrid Head
    head_path = os.path.join(model_dir, "hybrid_head.safetensors")
    if os.path.exists(head_path):
        print(f"Loading Hybrid Decision Head from {head_path}...")
        head_dict = load_file(head_path)
        compute_dtype = model.w_noul.weight.dtype
        model.w_noul.weight.data.copy_(head_dict["w_noul.weight"].to(device=device, dtype=compute_dtype))
        model.w_noul.bias.data.copy_(head_dict["w_noul.bias"].to(device=device, dtype=compute_dtype))
        model.w_score.weight.data.copy_(head_dict["w_score.weight"].to(device=device, dtype=compute_dtype))
        model.w_score.bias.data.copy_(head_dict["w_score.bias"].to(device=device, dtype=compute_dtype))
        model.q_proj.weight.data.copy_(head_dict["q_proj.weight"].to(device=device, dtype=compute_dtype))
        model.k_proj.weight.data.copy_(head_dict["k_proj.weight"].to(device=device, dtype=compute_dtype))
        model.tau_choice.data.copy_(head_dict["tau_choice"].to(device=device, dtype=torch.float32))
        model.tau_noul.data.copy_(head_dict["tau_noul"].to(device=device, dtype=torch.float32))
        model.tau_score.data.copy_(head_dict["tau_score"].to(device=device, dtype=torch.float32))

    collator = DecisionDataCollator(tokenizer=model.tokenizer, max_seq_length=2048)
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device.startswith("cuda")),
    )

    metrics = evaluate_model(
        model=model,
        val_dataloader=test_loader,
        loss_engine=loss_engine,
        device=device,
        max_eval_batches=None,
    )

    print("\n" + "=" * 60)
    print("HELD-OUT TEST EVALUATION METRICS:")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    return metrics


# ==============================================================================
# 7. MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="System One Test Dataset Automation & Verification")
    parser.add_argument("--train_path", type=str, default="decision_train_26k.jsonl", help="Path to training jsonl")
    parser.add_argument("--output_file", type=str, default="decision_test_2.6k.jsonl", help="Output test jsonl path")
    parser.add_argument("--seed", type=int, default=SEED_TEST, help="Random seed for test sampling")
    parser.add_argument("--verify_only", action="store_true", help="Only verify existing test file against train file")
    parser.add_argument("--evaluate_model_dir", type=str, default=None, help="Optional model directory to evaluate on test dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")

    args = parser.parse_args()

    if args.verify_only:
        verify_zero_overlap(args.train_path, args.output_file)
    else:
        build_test_dataset(args.train_path, args.output_file, seed=args.seed)

    if args.evaluate_model_dir:
        evaluate_test_dataset_with_model(args.evaluate_model_dir, args.output_file, device=args.device)


if __name__ == "__main__":
    main()
