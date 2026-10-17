"""Validate sample and slide separation before fitting or evaluating models."""
from itertools import combinations

from .stats import wsi_ids


def validate_splits(frames, allow_slide_overlap=False):
    ids, slides = {}, {}
    for name, frame in frames.items():
        if frame.empty or frame.sample_id.duplicated().any():
            raise ValueError(f"Empty split or duplicate sample IDs: {name}")
        ids[name] = set(frame.sample_id)
        slides[name] = set(wsi_ids(frame.sample_id))
    overlaps = {}
    for a, b in combinations(frames, 2):
        if ids[a] & ids[b]:
            raise ValueError(f"Sample overlap: {a}/{b}")
        overlaps[f"wsi_overlap_{a}_{b}"] = len(slides[a] & slides[b])
        if "patient_id" in frames[a] and "patient_id" in frames[b]:
            if frames[a].patient_id.isna().any() or frames[b].patient_id.isna().any():
                raise ValueError("Missing patient IDs")
            overlaps[f"patient_overlap_{a}_{b}"] = len(set(frames[a].patient_id.astype(str)) & set(frames[b].patient_id.astype(str)))
    if any(overlaps.values()) and not allow_slide_overlap:
        raise ValueError(f"Slide overlap: {overlaps}. Repair the split before inference.")
    return overlaps
