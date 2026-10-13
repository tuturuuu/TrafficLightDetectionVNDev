"""Half-open nominal feature-cell footprints, not neural receptive fields."""

from dataclasses import dataclass

import numpy as np


def feature_bounds(box, geometry, feature_hw, fallback=False):
    fh, fw = feature_hw
    if min(fh, fw) <= 0:
        raise ValueError('Feature dimensions must be positive')
    x1, y1, x2, y2 = geometry.transform_boxes([box])[0]
    ih, iw = geometry.input_hw
    mapped = np.array([x1 * fw / iw, y1 * fh / ih, x2 * fw / iw, y2 * fh / ih])
    c0, r0 = np.floor(mapped[:2]).astype(int)
    c1, r1 = np.ceil(mapped[2:]).astype(int)
    c0, c1 = np.clip([c0, c1], 0, fw)
    r0, r1 = np.clip([r0, r1], 0, fh)
    degenerate = bool(x2 <= x1 or y2 <= y1 or c1 <= c0 or r1 <= r0)
    if degenerate:
        if not fallback:
            return None, False
        c0 = int(np.clip(np.floor((mapped[0] + mapped[2]) / 2), 0, fw - 1))
        r0 = int(np.clip(np.floor((mapped[1] + mapped[3]) / 2), 0, fh - 1))
        c1, r1 = c0 + 1, r0 + 1
    return (slice(r0, r1), slice(c0, c1)), degenerate


def score_tiles(relevance, tiles, geometry):
    relevance = np.asarray(relevance)
    if relevance.ndim != 2 or not np.isfinite(relevance).all():
        raise ValueError('Expected finite 2D relevance probabilities')
    scores = np.empty(len(tiles), dtype=np.float32)
    fallbacks = 0
    for i, tile in enumerate(tiles):
        bounds, fallback = feature_bounds(tile[:4], geometry, relevance.shape, fallback=True)
        scores[i] = relevance[bounds].max()
        fallbacks += fallback
    return scores, fallbacks


def select_scores(scores, threshold, min_keep=0, force_keep_all=False):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1]')
    if isinstance(min_keep, bool) or not isinstance(min_keep, int) or min_keep < 0:
        raise ValueError('min_keep must be a nonnegative integer')
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError('Tile scores must be finite')
    keep = scores > threshold
    if force_keep_all:
        keep[:] = True
    elif keep.sum() < min(min_keep, len(scores)):
        keep[np.argsort(-scores, kind='stable')[:min_keep]] = True
    return keep


@dataclass
class SelectionResult:
    keep: np.ndarray
    scores: np.ndarray
    relevance_map: np.ndarray
    timings_ms: dict
    geometry: object
    fallback_count: int = 0
