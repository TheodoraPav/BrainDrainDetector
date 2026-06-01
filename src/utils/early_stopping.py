"""
Early stopping on validation Macro F1 (per LOSO fold).

Patience counts consecutive epochs without improvement after min_epochs.
Set patience=0 to disable (train all cfg.training.epochs).
"""


def early_stopping_should_stop(
    epochs_run: int,
    epochs_without_improvement: int,
    patience: int,
    min_epochs: int,
) -> bool:
    """True when training should stop for this fold."""
    if patience <= 0:
        return False
    if epochs_run < min_epochs:
        return False
    return epochs_without_improvement >= patience


def update_validation_score(
    score: float,
    best_score: float,
    epochs_without_improvement: int,
) -> tuple[float, int, bool]:
    """
    Updates early-stopping state after one validation epoch.

    Returns:
        new_best_score, new_epochs_without_improvement, improved
    """
    if score > best_score:
        return score, 0, True
    return best_score, epochs_without_improvement + 1, False
