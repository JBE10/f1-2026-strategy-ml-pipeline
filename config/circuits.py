# Circuit metadata for F1 2026 Simulation
# is_street: 1 if street/semi-street circuit, 0 otherwise
# overtaking_difficulty: 1 (Low), 2 (Medium), 3 (High)

CIRCUIT_METADATA = {
    'albert_park': {'is_street': 1, 'overtaking_difficulty': 2},
    'shanghai': {'is_street': 0, 'overtaking_difficulty': 1},
    'suzuka': {'is_street': 0, 'overtaking_difficulty': 2},
    'miami': {'is_street': 1, 'overtaking_difficulty': 2},
    'villeneuve': {'is_street': 1, 'overtaking_difficulty': 2},
    'monaco': {'is_street': 1, 'overtaking_difficulty': 3},
    'catalunya': {'is_street': 0, 'overtaking_difficulty': 2},
    'red_bull_ring': {'is_street': 0, 'overtaking_difficulty': 1},
    'silverstone': {'is_street': 0, 'overtaking_difficulty': 1},
    'spa': {'is_street': 0, 'overtaking_difficulty': 1},
    'hungaroring': {'is_street': 0, 'overtaking_difficulty': 3},
    'zandvoort': {'is_street': 0, 'overtaking_difficulty': 3},
    'monza': {'is_street': 0, 'overtaking_difficulty': 1},
    'madring': {'is_street': 1, 'overtaking_difficulty': 2},
    'baku': {'is_street': 1, 'overtaking_difficulty': 2},
    'marina_bay': {'is_street': 1, 'overtaking_difficulty': 3},
    'americas': {'is_street': 0, 'overtaking_difficulty': 1},
    'rodriguez': {'is_street': 0, 'overtaking_difficulty': 2},
    'interlagos': {'is_street': 0, 'overtaking_difficulty': 2},
    'vegas': {'is_street': 1, 'overtaking_difficulty': 1},
    'losail': {'is_street': 0, 'overtaking_difficulty': 2},
    'yas_marina': {'is_street': 0, 'overtaking_difficulty': 2},
    'jeddah': {'is_street': 1, 'overtaking_difficulty': 2},
    'imola': {'is_street': 0, 'overtaking_difficulty': 3},
    'sochi': {'is_street': 1, 'overtaking_difficulty': 3},
    'portimao': {'is_street': 0, 'overtaking_difficulty': 2},
    'mugello': {'is_street': 0, 'overtaking_difficulty': 3},
    'istanbul': {'is_street': 0, 'overtaking_difficulty': 1},
    'sepang': {'is_street': 0, 'overtaking_difficulty': 1},
    'bahrain': {'is_street': 0, 'overtaking_difficulty': 1},
}

def get_circuit_features(circuit_ref):
    """Returns a dict of features for a given circuit reference."""
    return CIRCUIT_METADATA.get(circuit_ref, {'is_street': 0, 'overtaking_difficulty': 2})
