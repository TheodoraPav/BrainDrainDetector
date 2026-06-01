"""
Shared label logic for BrainDrainDetector.

Ground truth uses only self-reported arousal and valence (1-5) — no emotion flags.

  classify_from_va   3-class rule: Optimal / Overloaded / Grey Zone
  merge_to_binary    Safe = {Optimal, Grey Zone}, Alarm = {Overloaded}
  derive_binary_from_va  Same rules on predicted (â, v̂) after regression_va
"""

# Internal 3-class constants (step 01 only)
OPTIMAL    = 0
OVERLOADED = 1
GREY_ZONE  = 2

# Binary alarm constants (training / evaluation)
SAFE  = 0
ALARM = 1


def classify_from_va(arousal: float, valence: float, cfg) -> int:
    """
    3-class rule from arousal and valence only.

    Overloaded: low valence AND high arousal
    Optimal:    high valence AND low arousal
    Grey Zone:  everything else

    Returns:
        0 = Optimal, 1 = Overloaded, 2 = Grey Zone
    """
    a = round(max(1.0, min(5.0, float(arousal))))
    v = round(max(1.0, min(5.0, float(valence))))

    if v <= cfg.overloaded_max_valence and a >= cfg.overloaded_min_arousal:
        return OVERLOADED

    if v >= cfg.optimal_min_valence and a <= cfg.optimal_max_arousal:
        return OPTIMAL

    return GREY_ZONE


def classify_window(row, cfg) -> int:
    """
    Step 01 entry point: reads arousal/valence from a self-annotation row.

    Emotion columns in the CSV are ignored.
    """
    return classify_from_va(int(row["arousal"]), int(row["valence"]), cfg)


def merge_to_binary(label: int) -> int:
    """
    Binary alarm mapping.

    Classes 0 (Optimal) and 2 (Grey Zone) map to Safe (0).
    Class 1 (Overloaded) maps to Alarm (1).
    """
    return ALARM if int(label) == OVERLOADED else SAFE


def derive_binary_from_va(arousal: float, valence: float, cfg) -> int:
    """Derives Safe/Alarm from predicted arousal and valence."""
    return merge_to_binary(classify_from_va(arousal, valence, cfg))
