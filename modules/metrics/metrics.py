import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve


# ==========================================================
# Cosine Similarity
# ==========================================================

def cosine_similarity(embedding1, embedding2):
    """
    Computes cosine similarity between two batches of embeddings.

    Args:
        embedding1 : (N, D)
        embedding2 : (N, D)

    Returns:
        similarity : (N,)
    """

    embedding1 = F.normalize(embedding1, dim=1)
    embedding2 = F.normalize(embedding2, dim=1)

    return F.cosine_similarity(
        embedding1,
        embedding2,
        dim=1
    )


# ==========================================================
# Create Verification Pairs
# ==========================================================

def create_verification_pairs(
    embeddings,
    labels
):
    """
    Generates all unique verification pairs.

    Returns
    -------
    scores : numpy array
    pair_labels : numpy array

    pair_label
        1 -> same speaker
        0 -> different speaker
    """

    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu()

    if torch.is_tensor(labels):
        labels = labels.cpu()

    scores = []
    pair_labels = []

    n = len(labels)

    for i in range(n):

        for j in range(i + 1, n):

            score = F.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).item()

            scores.append(score)

            pair_labels.append(
                int(labels[i] == labels[j])
            )

    return (
        np.asarray(scores),
        np.asarray(pair_labels)
    )


# ==========================================================
# Verification Accuracy
# ==========================================================

def verification_accuracy(
    embeddings,
    labels,
    threshold=0.5
):
    """
    Computes verification accuracy.

    Returns
    -------
    accuracy
    """

    scores, pair_labels = create_verification_pairs(
        embeddings,
        labels
    )

    predictions = (scores >= threshold).astype(int)

    accuracy = np.mean(
        predictions == pair_labels
    )

    return accuracy


# ==========================================================
# False Acceptance Rate
# ==========================================================

def false_acceptance_rate(
    scores,
    labels,
    threshold
):

    predictions = scores >= threshold

    false_accept = np.sum(
        (predictions == 1) &
        (labels == 0)
    )

    true_reject = np.sum(
        (predictions == 0) &
        (labels == 0)
    )

    denominator = false_accept + true_reject

    if denominator == 0:
        return 0.0

    return false_accept / denominator


# ==========================================================
# False Rejection Rate
# ==========================================================

def false_rejection_rate(
    scores,
    labels,
    threshold
):

    predictions = scores >= threshold

    false_reject = np.sum(
        (predictions == 0) &
        (labels == 1)
    )

    true_accept = np.sum(
        (predictions == 1) &
        (labels == 1)
    )

    denominator = false_reject + true_accept

    if denominator == 0:
        return 0.0

    return false_reject / denominator


# ==========================================================
# Equal Error Rate
# ==========================================================

def compute_eer(
    embeddings,
    labels
):
    """
    Computes Equal Error Rate.

    Returns
    -------
    eer
    """

    scores, pair_labels = create_verification_pairs(
        embeddings,
        labels
    )

    fpr, tpr, thresholds = roc_curve(
        pair_labels,
        scores,
        pos_label=1
    )

    fnr = 1 - tpr

    index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    eer = (fpr[index] + fnr[index]) / 2.0

    return eer


# ==========================================================
# Minimum Detection Cost Function
# ==========================================================

def compute_min_dcf(
    embeddings,
    labels,
    p_target=0.01,
    c_miss=1,
    c_fa=1
):

    scores, pair_labels = create_verification_pairs(
        embeddings,
        labels
    )

    fpr, tpr, thresholds = roc_curve(
        pair_labels,
        scores,
        pos_label=1
    )

    fnr = 1 - tpr

    dcf = (
        c_miss * fnr * p_target
        +
        c_fa * fpr * (1 - p_target)
    )

    return np.min(dcf)


# ==========================================================
# Complete Evaluation
# ==========================================================

def evaluate(
    embeddings,
    labels,
    threshold=0.5
):

    scores, pair_labels = create_verification_pairs(
        embeddings,
        labels
    )

    accuracy = verification_accuracy(
        embeddings,
        labels,
        threshold
    )

    eer = compute_eer(
        embeddings,
        labels
    )

    mindcf = compute_min_dcf(
        embeddings,
        labels
    )

    far = false_acceptance_rate(
        scores,
        pair_labels,
        threshold
    )

    frr = false_rejection_rate(
        scores,
        pair_labels,
        threshold
    )

    return {
        "Accuracy": accuracy,
        "EER": eer,
        "FAR": far,
        "FRR": frr,
        "minDCF": mindcf
    }