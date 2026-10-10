"""
download_and_curate_26k.py

Downloads and curates the 26,000-sample multi-domain dataset
for the Qwen2.5-1.5B decision model.

Output:
    decision_train_26k.jsonl
"""

import os
import json
import random
from datasets import load_dataset

SEED = 42
OUTPUT_FILE = "data/decision_train_26k.jsonl" if os.path.isdir("data") else "decision_train_26k.jsonl"

random.seed(SEED)


def format_choice_with_subsampling(
    state_dict: dict,
    question: str,
    all_options: list[str],
    ground_truth_idx: int,
    domain: str,
    max_distractors: int = 9,
    rng: random.Random = None,
) -> dict:
    """
    Subsamples distractors and dynamically shuffles candidates
    to prevent coordinate memorization and context length inflation.
    """
    if rng is None:
        rng = random

    correct_option = all_options[ground_truth_idx]
    distractors = [opt for i, opt in enumerate(all_options) if i != ground_truth_idx]

    # Subsample if candidate set is large (e.g., Banking77, CLINC150)
    if len(distractors) > max_distractors:
        selected_distractors = rng.sample(distractors, max_distractors)
    else:
        selected_distractors = list(distractors)

    # Dynamic permutation with indexed tracking to guarantee correct pointer
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
    """
    Normalizes SummEval 1..5 relevance rating to 0..4 class offset
    for zero-indexed rubric alignment with active_levels=5.
    """
    raw_score = round(ex["score"])
    # Clamp and convert 1..5 scale into 0..4 index
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
    """
    Normalizes STS-B semantic similarity (0..1 continuous score in sentence-transformers/stsb)
    to 0..5 integer rubric class indices with active_levels=6.
    """
    raw_val = ex["score"]
    # sentence-transformers/stsb normalizes scores to [0.0, 1.0]. Rescale to 0..5.
    scaled = raw_val * 5.0 if raw_val <= 1.0 else raw_val
    clamped = min(max(scaled, 0.0), 5.0)
    return {
        "task_type": "score",
        "domain": "semantic_similarity",
        "state": {
            "sentence1": ex["sentence1"],
            "sentence2": ex["sentence2"],
        },
        "question": (
            "Rate the semantic similarity of "
            "sentence1 and sentence2 from 0 to 5."
        ),
        "target_score": int(round(clamped)),
        "active_levels": 6,
    }


def add_dataset(name, dataset, formatter, limit, records_list, seed=SEED):
    """Shuffle, select samples, format them, and add to records."""

    print(f"\n[{name}]")
    print(f"Available samples: {len(dataset):,}")
    print(f"Selecting: {limit:,}")

    if len(dataset) < limit:
        raise ValueError(
            f"{name} has only {len(dataset)} samples, "
            f"but {limit} were requested."
        )

    if hasattr(dataset, "shuffle"):
        sub = dataset.shuffle(seed=seed).select(range(limit))
    else:
        rng = random.Random(seed)
        sub = rng.sample(dataset, limit)

    for ex in sub:
        records_list.append(formatter(ex))

    print(f"Added: {limit:,}")


def build_dataset():
    random.seed(SEED)
    records = []

    # ============================================================
    # 1. AGENTIC ROUTING & INTENT
    # Target: 6,500
    # ============================================================

    print("\n" + "=" * 60)
    print("1. AGENTIC ROUTING & INTENT")
    print("=" * 60)

    # Banking77 — 3,000
    banking77 = load_dataset(
        "mteb/banking77",
        split="train"
    )

    banking_label_map = {ex["label"]: ex["label_text"] for ex in banking77}
    banking_labels = [banking_label_map[i] for i in range(len(banking_label_map))]

    add_dataset(
        "Banking77",
        banking77,
        lambda ex: format_choice_with_subsampling(
            state_dict={"text": ex["text"]},
            question="Identify the primary customer service intent for this banking query.",
            all_options=banking_labels,
            ground_truth_idx=ex["label"],
            domain="finance_routing",
            max_distractors=9,
        ),
        3000,
        records,
    )

    # CLINC150 — 2,500
    clinc = load_dataset(
        "clinc/clinc_oos",
        "plus",
        split="train"
    )

    clinc_labels = clinc.features["intent"].names

    add_dataset(
        "CLINC150",
        clinc,
        lambda ex: format_choice_with_subsampling(
            state_dict={"query": ex["text"]},
            question="Route this utterance to the correct application intent or out-of-scope.",
            all_options=clinc_labels,
            ground_truth_idx=ex["intent"],
            domain="intent_routing",
            max_distractors=9,
        ),
        2500,
        records,
    )

    # xLAM / Apigen Tool Routing — 1,000
    apigen_raw = load_dataset(
        "argilla/apigen-function-calling",
        split="train"
    )

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

    tool_routing_ds = apigen_raw.filter(is_valid_tool_call)

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
        )

    add_dataset(
        "xLAM / Apigen Tool Routing",
        tool_routing_ds,
        format_tool_routing,
        1000,
        records,
    )

    # ============================================================
    # 2. FACTUALITY & GROUNDING
    # Target: 6,500
    # ============================================================

    print("\n" + "=" * 60)
    print("2. FACTUALITY & GROUNDING")
    print("=" * 60)

    # BoolQ — 2,500
    boolq = load_dataset(
        "google/boolq",
        split="train"
    )

    add_dataset(
        "BoolQ",
        boolq,
        lambda ex: {
            "task_type": "noul",
            "domain": "factuality",
            "state": {
                "passage": ex["passage"]
            },
            "question": (
                f"Based on the passage, is the answer to "
                f"'{ex['question']}' yes?"
            ),
            "target_bool": bool(ex["answer"])
        },
        2500,
        records,
    )

    # PAWS — 2,000
    paws = load_dataset(
        "google-research-datasets/paws",
        "labeled_final",
        split="train"
    )

    add_dataset(
        "PAWS",
        paws,
        lambda ex: {
            "task_type": "noul",
            "domain": "paraphrase",
            "state": {
                "sentence1": ex["sentence1"],
                "sentence2": ex["sentence2"]
            },
            "question": (
                "Are sentence1 and sentence2 "
                "semantically equivalent paraphrases?"
            ),
            "target_bool": bool(ex["label"] == 1)
        },
        2000,
        records,
    )

    # ANLI — 2,000
    anli = load_dataset(
        "facebook/anli",
        split="train_r1"
    )

    anli_options = [
        "entailment",
        "neutral",
        "contradiction"
    ]

    add_dataset(
        "ANLI",
        anli,
        lambda ex: format_choice_with_subsampling(
            state_dict={
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"]
            },
            question="Determine the relationship between the premise and hypothesis.",
            all_options=anli_options,
            ground_truth_idx=ex["label"],
            domain="natural_language_inference",
            max_distractors=9,
        ),
        2000,
        records,
    )

    # ============================================================
    # 3. SAFETY & GUARDRAILS
    # Target: 5,000
    # ============================================================

    print("\n" + "=" * 60)
    print("3. SAFETY & GUARDRAILS")
    print("=" * 60)

    # ToxicChat — 3,000
    toxic = load_dataset(
        "lmsys/toxic-chat",
        "toxicchat0124",
        split="train"
    )

    add_dataset(
        "ToxicChat",
        toxic,
        lambda ex: {
            "task_type": "noul",
            "domain": "safety",
            "state": {
                "user_message": ex["user_input"]
            },
            "question": (
                "Does this user message contain toxic, "
                "abusive, or harmful content?"
            ),
            "target_bool": bool(ex["toxicity"] == 1)
        },
        3000,
        records,
    )

    # Prompt Injection — full dataset (546)
    prompt_injection = load_dataset(
        "deepset/prompt-injections",
        split="train"
    )

    add_dataset(
        "Prompt Injections",
        prompt_injection,
        lambda ex: {
            "task_type": "noul",
            "domain": "security",
            "state": {
                "prompt": ex["text"]
            },
            "question": (
                "Is this input a prompt injection "
                "or jailbreak attempt?"
            ),
            "target_bool": bool(ex["label"] == 1)
        },
        len(prompt_injection),
        records,
    )

    # OpenAI Moderation Evaluation — 1,454 (balanced to reach 5,000 domain total)
    moderation = load_dataset(
        "mmathys/openai-moderation-api-evaluation",
        split="train"
    )

    MOD_POLICY_COLS = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]

    add_dataset(
        "OpenAI Moderation Evaluation",
        moderation,
        lambda ex: {
            "task_type": "noul",
            "domain": "content_moderation",
            "state": {
                "text": ex["prompt"]
            },
            "question": (
                "Does this content violate any "
                "applicable safety policy?"
            ),
            "target_bool": bool(any(ex.get(k, 0) == 1 for k in MOD_POLICY_COLS))
        },
        1454,
        records,
    )

    # ============================================================
    # 4. RUBRIC SCORING & EVALUATION
    # Target: 5,000
    # ============================================================

    print("\n" + "=" * 60)
    print("4. RUBRIC SCORING & EVALUATION")
    print("=" * 60)

    # HelpSteer2 — 3,000
    helpsteer = load_dataset(
        "nvidia/HelpSteer2",
        split="train"
    )

    add_dataset(
        "HelpSteer2",
        helpsteer,
        lambda ex: {
            "task_type": "score",
            "domain": "helpfulness_rubric",
            "state": {
                "prompt": ex["prompt"],
                "response": ex["response"]
            },
            "question": (
                "Rate the overall helpfulness of the "
                "response on a 0 to 4 rubric scale."
            ),
            "target_score": int(ex["helpfulness"]),
            "active_levels": 5
        },
        3000,
        records,
    )

    # STS-B — 1,200
    stsb = load_dataset(
        "sentence-transformers/stsb",
        split="train"
    )

    add_dataset(
        "STS-B",
        stsb,
        format_stsb,
        1200,
        records,
    )

    # SummEval — 800 (flattened from test split multi-summaries)
    summeval_raw = load_dataset(
        "mteb/summeval",
        split="test"
    )

    summeval_pairs = []
    for ex in summeval_raw:
        doc = ex.get("text", "")
        for summary, score in zip(
            ex.get("machine_summaries", []),
            ex.get("relevance", [])
        ):
            summeval_pairs.append({
                "doc": doc,
                "summary": summary,
                "score": score
            })

    add_dataset(
        "SummEval",
        summeval_pairs,
        format_summeval,
        800,
        records,
    )

    # ============================================================
    # 5. SPECIALIZED VERTICALS
    # Target: 3,000
    # ============================================================

    print("\n" + "=" * 60)
    print("5. SPECIALIZED VERTICALS")
    print("=" * 60)

    # LexGLUE — 1,200
    lexglue = load_dataset(
        "coastalcph/lex_glue",
        "unfair_tos",
        split="train"
    )

    add_dataset(
        "LexGLUE",
        lexglue,
        lambda ex: {
            "task_type": "noul",
            "domain": "legal",
            "state": {
                "text": ex["text"]
            },
            "question": (
                "Does this legal clause contain "
                "an unfair or potentially unlawful provision?"
            ),
            "target_bool": bool(len(ex["labels"]) > 0)
        },
        1200,
        records,
    )

    # PubMedQA — 1,000
    pubmed = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train"
    )

    med_options = [
        "yes",
        "no",
        "maybe"
    ]

    med_map = {
        "yes": 0,
        "no": 1,
        "maybe": 2
    }

    add_dataset(
        "PubMedQA",
        pubmed,
        lambda ex: format_choice_with_subsampling(
            state_dict={
                "context": " ".join(
                    ex["context"]["contexts"]
                )
            },
            question=ex["question"],
            all_options=med_options,
            ground_truth_idx=med_map.get(
                ex["final_decision"],
                2
            ),
            domain="biomedical",
            max_distractors=9,
        ),
        1000,
        records,
    )

    # Code Defect Detection — 800
    code_dataset = load_dataset(
        "google/code_x_glue_cc_defect_detection",
        split="train"
    )

    add_dataset(
        "Code Defect Detection",
        code_dataset,
        lambda ex: {
            "task_type": "noul",
            "domain": "code",
            "state": {
                "code": ex.get("func", "")
            },
            "question": (
                "Does this code contain a software "
                "defect or vulnerability?"
            ),
            "target_bool": bool(ex["target"])
        },
        800,
        records,
    )

    # ============================================================
    # FINAL SHUFFLE + VALIDATION
    # ============================================================

    print("\n" + "=" * 60)
    print("FINALIZING DATASET")
    print("=" * 60)

    random.shuffle(records)

    expected_count = 26000
    actual_count = len(records)

    print(f"\nExpected samples : {expected_count:,}")
    print(f"Actual samples   : {actual_count:,}")

    if actual_count != expected_count:
        raise RuntimeError(
            f"Dataset size mismatch! "
            f"Expected {expected_count}, got {actual_count}."
        )

    # Count task types
    task_counts = {}
    for record in records:
        task = record["task_type"]
        task_counts[task] = task_counts.get(task, 0) + 1

    print("\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        percentage = count / actual_count * 100
        print(
            f"  {task:10s}: "
            f"{count:6,} ({percentage:.2f}%)"
        )

    # Save JSONL
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"Saved dataset to: {OUTPUT_FILE}")
    print(f"Total records   : {actual_count:,}")


if __name__ == "__main__":
    build_dataset()
