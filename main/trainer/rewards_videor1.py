from __future__ import annotations

import re
from typing import Dict, List, Optional


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
FORMAT_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str:
    match = ANSWER_PATTERN.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_number(num_str: str) -> Optional[float]:
    try:
        return float(num_str.replace(",", ""))
    except Exception:
        return None


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_whitespace(reference).split()
    hyp_words = normalize_whitespace(hypothesis).split()
    m = len(ref_words)
    n = len(hyp_words)
    if m == 0:
        return 0.0 if n == 0 else 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / max(1, m)


def lcs_length(a: List[str], b: List[str]) -> int:
    m = len(a)
    n = len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_whitespace(reference).lower().split()
    hyp_tokens = normalize_whitespace(hypothesis).lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def format_reward(prediction: str) -> float:
    return 1.0 if FORMAT_PATTERN.fullmatch((prediction or "").strip()) else 0.0


def accuracy_reward(prediction: str, reference: str, problem_type: str) -> float:
    pred_answer = extract_answer(prediction)
    ref_answer = extract_answer(reference)

    if problem_type == "multiple choice":
        return 1.0 if normalize_whitespace(pred_answer).lower() == normalize_whitespace(ref_answer).lower() else 0.0

    if problem_type == "numerical":
        ref_has_decimal = "." in ref_answer or "," in ref_answer
        pred_has_decimal = "." in pred_answer or "," in pred_answer
        if ref_has_decimal != pred_has_decimal:
            return 0.0
        ref_num = normalize_number(ref_answer)
        pred_num = normalize_number(pred_answer)
        if ref_num is None or pred_num is None:
            return 0.0
        return 1.0 if round(ref_num, 2) == round(pred_num, 2) else 0.0

    if problem_type == "OCR":
        return max(0.0, min(1.0, 1.0 - word_error_rate(ref_answer, pred_answer)))

    if problem_type == "free-form":
        return max(0.0, min(1.0, rouge_l_f1(ref_answer, pred_answer)))

    if problem_type == "regression":
        ref_num = normalize_number(ref_answer)
        pred_num = normalize_number(pred_answer)
        if ref_num is None or pred_num is None:
            return 0.0
        rel_diff = (abs(pred_num - ref_num) + 1e-9) / (abs(ref_num) + 1e-9)
        rel_diff = max(0.0, min(1.0, rel_diff))
        return 1.0 - rel_diff

    return 1.0 if normalize_whitespace(pred_answer).lower() == normalize_whitespace(ref_answer).lower() else 0.0


def combined_reward(
    prediction: str,
    reference: str,
    meta: Optional[Dict[str, object]] = None,
    accuracy_weight: float = 1.0,
    format_weight: float = 1.0,
) -> Dict[str, float]:
    meta = meta or {}
    problem_type = str(meta.get("problem_type", "free-form"))
    acc = accuracy_reward(prediction, reference, problem_type)
    fmt = format_reward(prediction)
    total = accuracy_weight * acc + format_weight * fmt
    return {"accuracy": acc, "format": fmt, "total": total}
