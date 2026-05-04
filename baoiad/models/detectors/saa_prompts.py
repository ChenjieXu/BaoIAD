"""Prompt definitions for SAA/SAA+ (Segment Any Anomaly).

Reference: arXiv 2305.10724
Prompts ported from official SAA+ repo: SAA/prompts/mvtec_parameters.py

Key insight: SAA uses paired prompts ``(defect_prompt, filter_phrase)``.
The filter phrase is used to suppress detections of the object itself.
"""

from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# General anomaly prompts (used by both SAA and SAA+)
# Format: [prompt_template, filter_phrase]
# ---------------------------------------------------------------------------
GENERAL_ANOMALY_PROMPTS = [
    ['defect on {}', '{}'],
    ['damage on {}', '{}'],
    ['flaw on {}', '{}'],
]


def build_general_prompts(category: str) -> List[Tuple[str, str]]:
    """Build general prompts for a category.

    Args:
        category: Category name (e.g., 'bottle', 'cable').

    Returns:
        List of (prompt, filter_phrase) tuples.
    """
    return [(p.format(category), f.format(category)) for p, f in GENERAL_ANOMALY_PROMPTS]


# ---------------------------------------------------------------------------
# MVTec AD per-category manual prompts
# Format: [[prompt, filter_phrase], ...]
# ---------------------------------------------------------------------------
MVTEC_MANUAL_PROMPTS: Dict[str, List[List[str]]] = {
    'carpet': [
        ['black hole', 'carpet'],
        ['thread', 'carpet'],
        ['defect.', 'carpet'],
    ],
    'grid': [
        ['irregular pattern', 'grid'],
        ['defect.', 'grid'],
    ],
    'leather': [
        ['defect.', 'leather'],
    ],
    'tile': [
        ['defect.', 'tile'],
    ],
    'wood': [
        ['defect.', 'wood'],
    ],
    'bottle': [
        ['broken part. contamination. white broken.', 'bottle'],
    ],
    'cable': [
        ['crack. flawed golden wire. black hole.', 'cable'],
    ],
    'capsule': [
        ['white crack. hole.', 'capsule'],
    ],
    'hazelnut': [
        ['white print. crack. thread.', 'hazelnut'],
    ],
    'metal_nut': [
        ['blue defect. black defect. red defect. scratch.', 'nut'],
    ],
    'pill': [
        ['red defect. yellow defect. blue defect. crack. scratch.', 'pill'],
    ],
    'screw': [
        ['defect.', 'screw'],
    ],
    'toothbrush': [
        ['defect.', 'toothbrush'],
    ],
    'transistor': [
        ['defect.', 'transistor'],
    ],
    'zipper': [
        ['crack. broken leather.', 'zipper'],
    ],
}

# ---------------------------------------------------------------------------
# MVTec per-category property prompts
# Format: "the image of {cls} have {object_number} dissimilar {object}, with a maximum of {k_mask} anomaly. The anomaly would not exceed {defect_area_threshold} object area."
# ---------------------------------------------------------------------------
MVTEC_PROPERTY_PROMPTS: Dict[str, str] = {
    'carpet': 'the image of carpet have 1 dissimilar carpet, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'grid': 'the image of grid have 1 dissimilar grid, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'leather': 'the image of leather have 1 dissimilar leather, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'tile': 'the image of tile have 1 dissimilar tile, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'wood': 'the image of wood have 1 dissimilar wood, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'bottle': 'the image of bottle have 1 dissimilar bottle, with a maximum of 5 anomaly. The anomaly would not exceed 0.3 object area. ',
    'cable': 'the image of cable have 1 dissimilar cable, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'capsule': 'the image of capsule have 1 dissimilar capsule, with a maximum of 5 anomaly. The anomaly would not exceed 0.6 object area. ',
    'hazelnut': 'the image of hazelnut have 1 dissimilar hazelnut, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'metal_nut': 'the image of metal_nut have 1 dissimilar metal_nut, with a maximum of 5 anomaly. The anomaly would not exceed 1. object area. ',
    'pill': 'the image of pill have 1 dissimilar pill, with a maximum of 5 anomaly. The anomaly would not exceed 1. object area. ',
    'screw': 'the image of screw have 1 dissimilar screw, with a maximum of 5 anomaly. The anomaly would not exceed 0.1 object area. ',
    'toothbrush': 'the image of toothbrush have 1 dissimilar toothbrush, with a maximum of 5 anomaly. The anomaly would not exceed 0.5 object area. ',
    'transistor': 'the image of transistor have 1 dissimilar transistor, with a maximum of 5 anomaly. The anomaly would not exceed 1. object area. ',
    'zipper': 'the image of zipper have 1 dissimilar zipper, with a maximum of 5 anomaly. The anomaly would not exceed 0.5 object area. ',
}


def parse_property_prompt(property_prompt: str) -> dict:
    """Parse property prompt to extract parameters.

    Format: "the image of {cls} have {object_number} dissimilar {object}, with a maximum of {k_mask} anomaly. The anomaly would not exceed {defect_area_threshold} object area."

    Returns:
        dict with keys: object_prompt, object_number, k_mask, defect_area_threshold, object_max_area
    """
    parts = property_prompt.split(' ')
    # parts[7] is like 'bottle,' - need to strip trailing comma
    object_prompt = parts[7].rstrip(',')
    return {
        'object_prompt': object_prompt,  # e.g., 'bottle'
        'object_number': int(parts[5]),  # e.g., 1
        'k_mask': int(parts[12]),  # e.g., 5
        'defect_area_threshold': float(parts[19]),  # e.g., 0.3
        'object_max_area': 1.0 / int(parts[5]),  # e.g., 1.0
    }


# ---------------------------------------------------------------------------
# VisA per-category prompts
# ---------------------------------------------------------------------------
VISA_MANUAL_PROMPTS: Dict[str, List[List[str]]] = {
    'candle': [['damaged candle', 'candle'], ['stain on candle', 'candle']],
    'capsules': [['damaged capsule', 'capsule'], ['cracked capsule', 'capsule']],
    'cashew': [['damaged cashew', 'cashew'], ['crack on cashew', 'cashew']],
    'chewinggum': [['damaged chewing gum', 'chewing gum'], ['stain on chewing gum', 'chewing gum']],
    'fryum': [['damaged fryum', 'fryum'], ['crack on fryum', 'fryum']],
    'macaroni1': [['damaged macaroni', 'macaroni'], ['crack on macaroni', 'macaroni']],
    'macaroni2': [['damaged macaroni', 'macaroni'], ['crack on macaroni', 'macaroni']],
    'pcb1': [['damaged pcb', 'pcb'], ['scratch on pcb', 'pcb']],
    'pcb2': [['damaged pcb', 'pcb'], ['scratch on pcb', 'pcb']],
    'pcb3': [['damaged pcb', 'pcb'], ['scratch on pcb', 'pcb']],
    'pcb4': [['damaged pcb', 'pcb'], ['scratch on pcb', 'pcb']],
    'pipe_fryum': [['damaged pipe fryum', 'pipe fryum'], ['crack on pipe fryum', 'pipe fryum']],
}


def build_saa_prompts(
    cls_name: str,
    mode: str = 'saa',
    custom_prompts: Optional[Sequence[Sequence[str]]] = None,
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Assemble prompt list for SAA/SAA+.

    Args:
        cls_name: Category name (e.g. 'bottle', 'cable').
        mode: 'saa' for vanilla general prompts, 'saa+' for hybrid prompts.
        custom_prompts: If provided, override all automatic prompts with
            ``(defect_prompt, filter_phrase)`` pairs.

    Returns:
        prompts: List of (defect_prompt, filter_phrase) tuples.
        property_prompt: Property prompt string for SAA+ mode, None for SAA mode.
    """
    cls_lower = cls_name.lower().replace(' ', '_')

    # Start with general prompts
    prompts = build_general_prompts(cls_name)

    property_prompt = None

    if mode == 'saa+':
        # Add category-specific manual prompts
        if cls_lower in MVTEC_MANUAL_PROMPTS:
            prompts.extend((prompt, filter_phrase)
                           for prompt, filter_phrase in MVTEC_MANUAL_PROMPTS[cls_lower])
            property_prompt = MVTEC_PROPERTY_PROMPTS.get(cls_lower)
        elif cls_lower in VISA_MANUAL_PROMPTS:
            prompts.extend((prompt, filter_phrase)
                           for prompt, filter_phrase in VISA_MANUAL_PROMPTS[cls_lower])

    if custom_prompts is not None:
        prompts = [(p[0], p[1]) for p in custom_prompts]

    return prompts, property_prompt
