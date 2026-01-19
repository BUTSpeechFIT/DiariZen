# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import numpy as np
from itertools import combinations
from scipy.special import softmax


def scp2path(scp_file):
    """ return path list """
    lines = [line.strip().split()[1] for line in open(scp_file)]
    return lines


def build_powerset_mapping(num_classes, max_set_size):
    """
    Build powerset mapping matrix.

    Returns
    -------
    mapping : (num_powerset_classes, num_classes) np.ndarray
        mapping[i, j] == 1 if jth class is in ith powerset class
    """
    num_powerset_classes = sum(
        int(np.math.comb(num_classes, i)) for i in range(max_set_size + 1)
    )
    mapping = np.zeros((num_powerset_classes, num_classes))
    powerset_k = 0
    for set_size in range(max_set_size + 1):
        for current_set in combinations(range(num_classes), set_size):
            mapping[powerset_k, list(current_set)] = 1
            powerset_k += 1
    return mapping


def reduce_powerset_to_top_speakers(
    logits_powerset: np.ndarray,
    original_mapping: np.ndarray,
    num_speakers_to_keep: int = 2,
    max_set_size: int = 2,
):
    """
    Reduce powerset logits from N speakers to top-K most active speakers.

    Parameters
    ----------
    logits_powerset : (num_frames, num_powerset_classes) np.ndarray
        Powerset logits from the model (e.g., 11 classes for 4 speakers)
    original_mapping : (num_powerset_classes, num_classes) np.ndarray
        Original powerset mapping matrix from pyannote
    num_speakers_to_keep : int
        Number of top speakers to keep (default: 2)
    max_set_size : int
        Maximum set size for the new powerset (default: 2)

    Returns
    -------
    reduced_probs : (num_frames, new_num_powerset_classes) np.ndarray
        Reduced powerset probabilities (e.g., 4 classes for 2 speakers)
    top_speakers : np.ndarray
        Indices of the top-K speakers that were kept
    reduced_mapping : (new_num_powerset_classes, num_speakers_to_keep) np.ndarray
        New powerset mapping for the reduced speakers
    """
    # Convert logits to probabilities
    probs_powerset = softmax(logits_powerset, axis=-1)

    # Convert to multilabel to detect speaker activity
    probs_multilabel = probs_powerset @ original_mapping  # (num_frames, num_speakers)

    # Sum activity across all frames to find most active speakers
    activity_per_speaker = probs_multilabel.sum(axis=0)  # (num_speakers,)

    # Get indices of top-K most active speakers
    top_speakers = np.argsort(activity_per_speaker)[-num_speakers_to_keep:]
    top_speakers = np.sort(top_speakers)  # Keep in order for consistency

    # Build mapping for original powerset classes to new reduced classes
    # New powerset has classes: {}, {0}, {1}, {0,1} for 2 speakers
    reduced_mapping = build_powerset_mapping(num_speakers_to_keep, min(max_set_size, num_speakers_to_keep))
    num_reduced_classes = reduced_mapping.shape[0]

    # For each original powerset class, determine which reduced class it maps to
    # by checking which of the top speakers are active
    reduced_probs = np.zeros((logits_powerset.shape[0], num_reduced_classes))

    for orig_class_idx in range(original_mapping.shape[0]):
        # Get which speakers are active in this original class
        active_speakers = np.where(original_mapping[orig_class_idx] == 1)[0]

        # Filter to only top speakers
        active_in_top = [s for s in active_speakers if s in top_speakers]

        # Skip if more speakers than max_set_size (shouldn't happen with max_set_size=2)
        if len(active_in_top) > max_set_size:
            continue

        # Map to new speaker indices (0, 1, ... num_speakers_to_keep-1)
        new_active = [np.where(top_speakers == s)[0][0] for s in active_in_top]

        # Find which reduced class has exactly these speakers active
        for reduced_class_idx in range(num_reduced_classes):
            reduced_active = np.where(reduced_mapping[reduced_class_idx] == 1)[0]
            if set(new_active) == set(reduced_active):
                reduced_probs[:, reduced_class_idx] += probs_powerset[:, orig_class_idx]
                break

    # Normalize to ensure probabilities sum to 1
    row_sums = reduced_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    reduced_probs = reduced_probs / row_sums

    return reduced_probs, top_speakers, reduced_mapping


def expand_multilabel_to_original_speakers(
    multilabel_reduced: np.ndarray,
    top_speakers: np.ndarray,
    num_original_speakers: int,
):
    """
    Expand reduced multilabel predictions back to original speaker indices.

    Parameters
    ----------
    multilabel_reduced : (num_frames, num_speakers_kept) np.ndarray
        Multilabel predictions for reduced speakers
    top_speakers : np.ndarray
        Indices of the top speakers in original space
    num_original_speakers : int
        Number of speakers in original space

    Returns
    -------
    multilabel_original : (num_frames, num_original_speakers) np.ndarray
        Multilabel predictions in original speaker space
    """
    multilabel_original = np.zeros((multilabel_reduced.shape[0], num_original_speakers))
    for i, orig_idx in enumerate(top_speakers):
        multilabel_original[:, orig_idx] = multilabel_reduced[:, i]
    return multilabel_original
