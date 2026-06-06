import json
import re
import string
import sys
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        if text == "a":
            return text
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0, 0, 0)
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def eval_answer(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall


def update_answer(metrics, prediction, gold):
    if not isinstance(gold, list):
        gold = [gold]

    em = max(exact_match_score(prediction, answer) for answer in gold)
    f1, prec, recall = max((f1_score(prediction, answer) for answer in gold), key=lambda x: x[0])

    metrics["em"] += float(em)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall
    return em, f1, prec, recall


def eval(prediction, gold_file):
    with open(gold_file, encoding="utf-8") as f:
        gold = json.load(f)

    metrics = {"em": 0, "f1": 0, "prec": 0, "recall": 0}
    num_answers = 0
    for dp in gold:
        cur_id = str(dp["test_id"])
        if cur_id not in prediction["answer"]:
            continue
        num_answers += 1
        update_answer(metrics, prediction["answer"][cur_id], dp["answer"])

    if num_answers == 0:
        return {key: 0.0 for key in metrics}

    for key in metrics:
        metrics[key] = round(metrics[key] / num_answers * 100, 2)
    return metrics


def eval_detailed(prediction, gold_file):
    with open(gold_file, encoding="utf-8") as f:
        gold = json.load(f)

    metrics = {"em": 0, "f1": 0, "prec": 0, "recall": 0}
    metrics_detailed = {}
    num_answers = 0
    for dp in gold:
        cur_id = str(dp["test_id"])
        if cur_id not in prediction["answer"]:
            continue
        num_answers += 1
        metrics_detailed[cur_id] = update_answer(metrics, prediction["answer"][cur_id], dp["answer"])

    if num_answers == 0:
        return {"metrics": {key: 0.0 for key in metrics}, "detailed": metrics_detailed}

    for key in metrics:
        metrics[key] = metrics[key] / num_answers * 100
    return {"metrics": metrics, "detailed": metrics_detailed}


if __name__ == "__main__":
    eval(sys.argv[1], sys.argv[2])
