import torch
import torch.nn.functional as F

def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (L1 Loss)
    """
    return torch.abs(pred - target).mean()


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error (L2 Loss)
    """
    return ((pred - target) ** 2).mean()


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Binary Focal Loss
    pred: logits (before sigmoid)
    target: binary label (0 or 1)
    """
    p = torch.sigmoid(pred)
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = p * target + (1 - p) * (1 - target)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Lovasz hinge loss for binary segmentation (flattened version)
    logits: [P], raw predictions
    labels: [P], binary ground truth (0 or 1)
    """
    if logits.numel() == 0:
        return logits.sum() * 0.

    signs = 2. * labels - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    labels_sorted = labels[perm]
    grad = lovasz_grad(labels_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Computes gradient of Lovasz extension with respect to sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Lovasz hinge loss (binary)
    pred: logits (before sigmoid), shape [B, ...]
    target: binary ground truth, shape [B, ...]
    """
    losses = []
    for logit, label in zip(pred, target):
        logit_flat = logit.view(-1)
        label_flat = label.view(-1)
        losses.append(lovasz_hinge_flat(logit_flat, label_flat))
    return torch.stack(losses).mean()

def binary_cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary Cross Entropy (with logits)
    pred: logits (before sigmoid), shape [B, ...]
    target: binary labels (0 or 1), same shape as pred
    """
    return F.binary_cross_entropy_with_logits(pred, target)


def cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Multiclass Cross Entropy
    pred: logits, shape [B, C, ...] (e.g., [B, C] or [B, C, H, W])
    target: class indices, shape [B, ...] (e.g., [B] or [B, H, W])
    """
    return F.cross_entropy(pred, target)
