import numpy as np
from typing import List, Dict, Any

def accuracy(outputs, targets):
    """
    Compute accuracy.
    
    Args:
        outputs: Model outputs
        targets: Ground truth targets
        
    Returns:
        Accuracy
    """
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100. * correct / total

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
    return np.mean([max(0, accuracies[i] - accuracies[current_step]) for i in range(current_step)])

# Note: The following metrics functions were removed as they were unused in the codebase:
# - backward_transfer: Computes the difference between current accuracy on previous tasks and original accuracy
# - forward_transfer: Computes the difference between accuracy on future tasks and random baseline
# - average_accuracy: Computes the mean accuracy across all steps
#
# If these metrics are needed in the future, they can be reimplemented based on the forgetting measure pattern.
