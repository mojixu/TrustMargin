import json
import logging
from pathlib import Path

from tqdm import tqdm

from evaluate import eval as evaluate_predictions
from evaluate import exact_match_score, f1_score

logger = logging.getLogger(__name__)


DATASET_NAMES = [
    "2wikimultihopqa",
    "complexwebquestions",
    "hotpotqa",
    "popqa",
]


def _score_prediction(prediction, gold):
    gold_answers = gold if isinstance(gold, list) else [gold]
    em = max(exact_match_score(prediction, answer) for answer in gold_answers)
    f1, _, _ = max((f1_score(prediction, answer) for answer in gold_answers), key=lambda x: x[0])
    return {"em": int(em), "f1": round(float(f1), 4)}


class QADataset:
    def __init__(
        self,
        name,
        data_file,
        data_loaded=None,
        require_passages=False,
        topk=None,
    ):
        if name not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset '{name}'. Expected one of {DATASET_NAMES}.")

        self.name = name
        self.gold_file = str(data_file)
        self.require_passages = require_passages
        self.topk = topk

        if data_loaded is None:
            logger.info("Loading %s dataset from %s.", name, data_file)
            with open(data_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info("Dataset loaded: %d samples.", len(self.data))
        else:
            self.data = data_loaded

        if require_passages:
            missing = [str(datum.get("test_id")) for datum in self.data if "passages" not in datum]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(
                    f"{self.gold_file} is missing passages for BM25-RAG samples: {preview}. "
                    "Run src/retrieve.py first."
                )

    def get_context(self, datum):
        passages = list(datum.get("passages", []))
        if self.topk is not None:
            passages = passages[: self.topk]
        return [f"Passage {idx + 1}: {passage}" for idx, passage in enumerate(passages)]

    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None or max_num_samples < 0:
            return self
        return QADataset(
            name=self.name,
            data_file=self.gold_file,
            data_loaded=self.data[:max_num_samples],
            require_passages=self.require_passages,
            topk=self.topk,
        )

    def inference(self, creator):
        predictions = {}
        records = []

        for datum in tqdm(self.data, desc=self.name):
            test_id = str(datum["test_id"])
            question = datum["question"]
            context = self.get_context(datum)
            materials = {
                "contents": {
                    "context": context,
                    "test_id": test_id,
                    "dataset": self.name,
                    # Gold is only exposed for post-hoc debug fields such as candidate oracle.
                    # Method scoring, prompting, routing, and selection must not read it.
                    "gold_answer_for_debug": datum["answer"],
                }
            }

            modelbox = creator.build(materials)
            prediction = modelbox.predict(question)
            raw_output = getattr(modelbox, "last_raw_output", prediction)
            creator.recover()

            predictions[test_id] = prediction
            record = {
                "test_id": test_id,
                "question": question,
                "answer": datum["answer"],
                "prediction": prediction,
                "raw_output": raw_output,
                "score": _score_prediction(prediction, datum["answer"]),
            }
            if context:
                record["passages"] = list(datum.get("passages", []))[: self.topk]
            method_debug = getattr(modelbox, "last_debug", None)
            if method_debug:
                record["method_debug"] = method_debug
            records.append(record)

        return {
            "predictions": {"answer": predictions},
            "records": records,
        }

    def evaluate(self, predictions):
        return evaluate_predictions(predictions, self.gold_file)


def load_dataset(
    name,
    split="dev",
    data_root="data",
    max_num_samples=None,
    require_passages=False,
    topk=None,
):
    data_file = Path(data_root) / name / f"{split}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    return QADataset(
        name=name,
        data_file=data_file,
        require_passages=require_passages,
        topk=topk,
    ).derive_trunc_dataset(max_num_samples)


def resolve_dataset_names(dataset, datasets=None):
    if datasets:
        names = datasets
    elif dataset == "all":
        names = DATASET_NAMES
    else:
        names = [dataset]

    unknown = [name for name in names if name not in DATASET_NAMES]
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}. Expected one of {DATASET_NAMES}.")
    return names
