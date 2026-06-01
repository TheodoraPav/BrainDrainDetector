"""
Shared label logic for BrainDrainDetector.

  classify_window    Internal 3-class rule (V, A, emotion flags) used in step 01
                     to assign each window before collapsing to binary.

  merge_to_binary    Safe = {Optimal, Grey Zone}, Alarm = {Overloaded}

  derive_binary_from_va  Post-training: derive Safe/Alarm from predicted A/V
                         via va_only rules (no emotion flags).
"""

# Internal 3-class constants (step 01 only)
OPTIMAL    = 0
OVERLOADED = 1
GREY_ZONE  = 2

# Binary alarm constants (training / evaluation)
SAFE  = 0
ALARM = 1


def classify_window(row, cfg) -> int:
    """
    Full 3-class rule: uses V, A, and categorical emotion flags.

    Used internally in step 01; the stored training label is always binary
    (see merge_to_binary).

    Returns:
        0 = Optimal, 1 = Overloaded, 2 = Grey Zone
    """
    valence = int(row["valence"])
    arousal = int(row["arousal"])

    def is_marked(col_name: str) -> bool:
        return str(row.get(col_name, "")).strip().lower() == "x"

    negative_emotion_active = any(
        is_marked(emotion) for emotion in cfg.overloaded_negative_emotions
    )
    if valence <= cfg.overloaded_max_valence and (negative_emotion_active or arousal >= cfg.overloaded_min_arousal):
        return OVERLOADED

    forbidden_active = any(is_marked(emotion) for emotion in cfg.optimal_forbidden_emotions)
    if (
        is_marked(cfg.optimal_required_emotion)
        and valence >= cfg.optimal_min_valence
        and not forbidden_active
    ):
        return OPTIMAL

    return GREY_ZONE


def _derive_3class_from_va(arousal: float, valence: float, cfg) -> int:
    """VA-only 3-class rule (internal helper for derive_binary_from_va)."""
    a = round(max(1.0, min(5.0, float(arousal))))
    v = round(max(1.0, min(5.0, float(valence))))

    if v <= cfg.va_only_overloaded_max_valence and a >= cfg.va_only_overloaded_min_arousal:
        return OVERLOADED

    if v >= cfg.va_only_optimal_min_valence and a <= cfg.va_only_optimal_max_arousal:
        return OPTIMAL

    return GREY_ZONE


def merge_to_binary(label: int) -> int:
    """
    Binary alarm mapping.

    Accepts either a legacy 3-class label (0/1/2) or an already-binary label.
    Classes 0 (Optimal) and 2 (Grey Zone) map to Safe (0).
    Class 1 (Overloaded) maps to Alarm (1).
    """
    return ALARM if int(label) == OVERLOADED else SAFE


def derive_binary_from_va(arousal: float, valence: float, cfg) -> int:
    """Derives Safe/Alarm from predicted arousal and valence (va_only rules)."""
    return merge_to_binary(_derive_3class_from_va(arousal, valence, cfg))
