"""
tests/test_banking77_labels.py

Regression test verifying that Banking77 labels are indexed canonically
by integer label ID 0..76.
"""

from datasets import load_dataset


def test_banking77_label_ordering():
    banking_test = load_dataset("mteb/banking77", split="test")

    # Correct canonical mapping:
    banking_label_map = {ex["label"]: ex["label_text"] for ex in banking_test}
    banking_labels = [banking_label_map[i] for i in range(len(banking_label_map))]

    assert len(banking_labels) == 77
    assert banking_labels[0] == "activate_my_card"
    assert banking_labels[1] == "age_limit"
    assert banking_labels[2] == "apple_pay_or_google_pay"
    assert banking_labels[76] == "wrong_exchange_rate_for_cash_withdrawal"

    # Verify every example in test matches its canonical label
    for ex in list(banking_test)[:100]:
        assert banking_labels[ex["label"]] == ex["label_text"]
