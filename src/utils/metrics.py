from typing import List

import numpy as np


def forgetting(accuracies: List[float], current_step: int) -> float:
    """
    Compute forgetting measure.

    Args:
        accuracies: List of accuracies for each step
        current_step: Current step

    Returns:
        Forgetting measure
    """
    if current_step == 0:
        return 0.0

    # Compute forgetting as the difference between the maximum accuracy
    # achieved on previous tasks and the current accuracy on those tasks
    return np.mean(
        [max(0, accuracies[i] - accuracies[current_step]) for i in range(current_step)]
    )
