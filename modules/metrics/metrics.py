import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve


# ---------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------

def cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1 : Tensor (B, D)
        embedding2 : Tensor (B, D)

    Returns:
        similarity : Tensor (B,)
    """

    return F.cosine_similarity(
        embedding1,
        embedding2,
        dim=1
    )


# ---------------------------------------------------------
# Verification Accuracy
# ---------------------------------------------------------

def verification_accuracy(scores,
                          labels,
                          threshold=0.5):
    """
    Args:
        scores : numpy array
        labels : numpy array
                 1 -> same speaker
                 0 -> different speaker

    Returns:
        accuracy
    """

    predictions = (scores >= threshold).astype(int)

    accuracy = np.mean(predictions == labels)

    return accuracy


# ---------------------------------------------------------
# FAR
# ---------------------------------------------------------

def false_acceptance_rate(scores,
                          labels,
                          threshold):
    """
    FAR = FP / (FP + TN)
    """

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
        return 0

    return false_accept / denominator


# ---------------------------------------------------------
# FRR
# ---------------------------------------------------------

def false_rejection_rate(scores,
                         labels,
                         threshold):
    """
    FRR = FN / (TP + FN)
    """

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
        return 0

    return false_reject / denominator


# ---------------------------------------------------------
# Equal Error Rate (EER)
# ---------------------------------------------------------

def compute_eer(scores,
                labels):
    """
    Compute Equal Error Rate.

    Args:
        scores : numpy array
        labels : numpy array
                 1 = genuine
                 0 = impostor

    Returns:
        eer
        threshold
    """

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1
    )

    fnr = 1 - tpr

    index = np.nanargmin(np.absolute(fnr - fpr))

    eer = (fpr[index] + fnr[index]) / 2

    threshold = thresholds[index]

    return eer, threshold


# ---------------------------------------------------------
# minDCF
# ---------------------------------------------------------

def compute_min_dcf(scores,
                    labels,
                    p_target=0.01,
                    c_miss=1,
                    c_fa=1):
    """
    Compute minimum Detection Cost Function.

    Returns:
        minDCF
        threshold
    """

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1
    )

    fnr = 1 - tpr

    dcf = (
        c_miss * fnr * p_target +
        c_fa * fpr * (1 - p_target)
    )

    index = np.argmin(dcf)

    return dcf[index], thresholds[index]


# ---------------------------------------------------------
# Evaluate All Metrics
# ---------------------------------------------------------

def evaluate(scores,
             labels):
    """
    Complete evaluation.

    Returns dictionary.
    """

    eer, eer_threshold = compute_eer(
        scores,
        labels
    )

    min_dcf, dcf_threshold = compute_min_dcf(
        scores,
        labels
    )

    accuracy = verification_accuracy(
        scores,
        labels,
        eer_threshold
    )

    far = false_acceptance_rate(
        scores,
        labels,
        eer_threshold
    )

    frr = false_rejection_rate(
        scores,
        labels,
        eer_threshold
    )

    return {
        "Accuracy": accuracy,
        "EER": eer,
        "Threshold": eer_threshold,
        "FAR": far,
        "FRR": frr,
        "minDCF": min_dcf,
        "minDCF Threshold": dcf_threshold
    }
