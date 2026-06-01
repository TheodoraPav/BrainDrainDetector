"""
Shared label logic for BrainDrainDetector.

Three functions cover the full label hierarchy:

  classify_window       Full 3-class rule using V, A, and categorical emotion
                        flags from self-annotation CSVs. Used in step 01 to
                        produce labels.csv.

  derive_3class_from_va VA-only 3-class rule that uses only predicted arousal
                        and valence. Applied post-training to derive an
                        operational class from regression outputs.

  merge_to_binary       Collapses the 3-class label to binary alarm:
                        Safe  = {Optimal, Grey Zone}
                        Alarm = {Overloaded}
"""

# ── 3-class constants ──────────────────────────────────────────────────────────
OPTIMAL    = 0
OVERLOADED = 1
GREY_ZONE  = 2

# ── Binary alarm constants ─────────────────────────────────────────────────────
SAFE  = 0
ALARM = 1


def classify_window(row, cfg) -> int:
    """
    Full 3-class rule: uses V, A, and categorical emotion flags.

    Args:
        row: one row from a P{N}.self.csv (valence, arousal, emotion columns)
        cfg: the 'labels' section of the YAML config

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


def derive_3class_from_va(arousal: float, valence: float, cfg) -> int:
    """
    VA-only 3-class rule: uses only arousal and valence (no emotion flags).

    Applied post-training to derive a predicted operational class from
    regression model outputs. The inputs are clamped to [1, 5] and rounded
    before the rule is evaluated.

    Args:
        arousal: predicted or true arousal value (continuous, ~1-5)
        valence: predicted or true valence value (continuous, ~1-5)
        cfg:     the 'labels' section of the YAML config

    Returns:
        0 = Optimal, 1 = Overloaded, 2 = Grey Zone
    """
    a = round(max(1.0, min(5.0, float(arousal))))
    v = round(max(1.0, min(5.0, float(valence))))

    if v <= cfg.va_only_overloaded_max_valence and a >= cfg.va_only_overloaded_min_arousal:
        return OVERLOADED

    if v >= cfg.va_only_optimal_min_valence and a <= cfg.va_only_optimal_max_arousal:
        return OPTIMAL

    return GREY_ZONE


def merge_to_binary(label: int) -> int:
    """
    Binary alarm mapping for deployment.

    Classes 0 (Optimal) and 2 (Grey Zone) map to Safe (0).
    Class 1 (Overloaded) maps to Alarm (1).
    """
    return ALARM if label == OVERLOADED else SAFE
