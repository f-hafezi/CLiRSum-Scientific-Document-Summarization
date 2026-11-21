import numpy as np
import evaluate

def load_metrics():
    return {
        "rouge": evaluate.load("rouge"),
        "bleu": evaluate.load("bleu"),
        "meteor": evaluate.load("meteor"),
        "bertscore": evaluate.load("bertscore"),
    }

def compute_task_metrics(predictions, references, metric_objects):
    predictions = [text.strip() for text in predictions]
    references = [text.strip() for text in references]

    if not predictions:
        return None

    results = {
        "rouge": metric_objects["rouge"].compute(
            predictions=predictions, references=references
        ),
        "bleu": metric_objects["bleu"].compute(
            predictions=predictions, references=[[ref] for ref in references]
        ),
        "meteor": metric_objects["meteor"].compute(
            predictions=predictions, references=references
        ),
        "bertscore": metric_objects["bertscore"].compute(
            predictions=predictions, references=references, lang="en"
        ),
    }

    for key in ("precision", "recall", "f1"):
        results["bertscore"][key] = float(
            np.mean(results["bertscore"][key])
        )
    return results
