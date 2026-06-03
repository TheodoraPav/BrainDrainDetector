"""
Shared label logic for BrainDrainDetector.

Ground truth uses only self-reported arousal and valence (1-5) — no emotion flags.

  classify_from_va   VA zones: Optimal / Overloaded / Grey (intermediate)
  merge_to_binary    Safe = {Optimal, Grey}, Alarm = {Overloaded} — training target
  derive_binary_from_va  Same rules on predicted (â, v̂) after regression_va
"""

# VA-zone constants (intermediate; merged to Safe/Alarm before training)
OPTIMAL    = 0
OVERLOADED = 1
GREY_ZONE  = 2

# Binary alarm constants (training / evaluation)
SAFE  = 0
ALARM = 1


def classify_from_va(arousal: float, valence: float, cfg) -> int:
    """
    VA-zone rule from arousal and valence only (merged to binary for training).

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


def _hl_value_sets(labels_cfg) -> tuple[set[int], set[int]]:
    """Returns (low_set, high_set) of rounded Likert levels, default Low=1–3 High=4–5."""
    hl = getattr(labels_cfg, "va_high_low", None)
    if hl is not None:
        low = getattr(hl, "low_values", None)
        high = getattr(hl, "high_values", None)
        if low is not None and high is not None:
            return set(int(x) for x in low), set(int(x) for x in high)
    return {1, 2, 3}, {4, 5}


def _likert_to_high_low(value: float, labels_cfg) -> int:
    """
    Binary target: 1 = High (4–5), 0 = Low (1–3) on rounded 1–5 Likert scale.
    """
    v = round(max(1.0, min(5.0, float(value))))
    low_set, high_set = _hl_value_sets(labels_cfg)
    if v in high_set:
        return 1
    if v in low_set:
        return 0
    raise ValueError(f"Likert value {v} not in Low {low_set} or High {high_set}")


def arousal_to_high_low(arousal: float, labels_cfg) -> int:
    """Binary arousal: High = 4–5, Low = 1–3."""
    return _likert_to_high_low(arousal, labels_cfg)


def valence_to_high_low(valence: float, labels_cfg) -> int:
    """Binary valence: High = 4–5, Low = 1–3."""
    return _likert_to_high_low(valence, labels_cfg)


def high_low_to_va_proxy(arousal_hl: int, valence_hl: int, cfg) -> tuple[float, float]:
    """
    Maps predicted High/Low classes to representative (A, V) for VA alarm rules.
    """
    labels = _labels_section(cfg)
    a_hi = float(labels.overloaded_min_arousal)
    a_lo = float(labels.optimal_max_arousal)
    v_hi = float(labels.optimal_min_valence)
    v_lo = float(labels.overloaded_max_valence)
    arousal = a_hi if int(arousal_hl) == 1 else a_lo
    valence = v_hi if int(valence_hl) == 1 else v_lo
    return arousal, valence


def _labels_section(cfg):
    return cfg.labels if hasattr(cfg, "labels") else cfg


def derive_alarm_from_high_low(arousal_hl: int, valence_hl: int, cfg) -> int:
    """Overload alarm from High/Low predictions via standard VA rules."""
    a, v = high_low_to_va_proxy(arousal_hl, valence_hl, cfg)
    return derive_binary_from_va(a, v, _labels_section(cfg))
