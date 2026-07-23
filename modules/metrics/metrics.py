import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve

# Cosine Similarity
def cosine_similarity(embedding1, embedding2):
    embedding1 = F.normalize(embedding1, dim=1)
    embedding2 = F.normalize(embedding2, dim=1)

    return F.cosine_similarity(
        embedding1,
        embedding2,
        dim=1
    )


# Verification Pairs (Vectorized)
def create_verification_pairs(
    embeddings,
    labels
):
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu()

    if torch.is_tensor(labels):
        labels = labels.cpu()

    n = len(labels)

    #  Normalize embeddings 
    embeddings = F.normalize(embeddings, p=2, dim=1)

    #Compute similarity matrix for all pairs at once (N x N)
    sim_matrix = torch.mm(embeddings, embeddings.T)

    # Compute label match matrix at once (N x N)
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

    # Extract upper triangular indices (unique pairs, excluding self)
    row_indices, col_indices = torch.triu_indices(n, n, offset=1)

    # Index the matrices to get flat arrays
    scores = sim_matrix[row_indices, col_indices].numpy()
    pair_labels = label_matrix[row_indices, col_indices].int().numpy()

    return scores, pair_labels


# Verification Accuracy
def verification_accuracy(
    embeddings,
    labels,
    threshold=0.5
):
    scores, pair_labels = create_verification_pairs(
        embeddings,
        labels
    )

    predictions = (scores >= threshold).astype(int)

    accuracy = np.mean(
        predictions == pair_labels
    )

    return accuracy

# False Acceptance Rate
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

# False Rejection Rate
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

# Equal Error Rate
def compute_eer(
    embeddings,
    labels
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

    index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    eer = (fpr[index] + fnr[index]) / 2.0

    return eer


# Minimum Detection Cost Function
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

# Complete Evaluation
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