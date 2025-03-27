import dspy
from typing import List, Callable

def run_evaluation(eval_set: List[dspy.Example], metric: Callable, signature: dspy.Signature) -> float:
    evaluate = dspy.Evaluate(devset=eval_set, metric=metric, num_threads=4,
                         display_progress=True, display_table=5)
    return evaluate(signature)