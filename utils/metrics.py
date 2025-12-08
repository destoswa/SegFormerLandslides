import numpy as np
import torch


def compute_iou(preds, labels, num_classes):
    ious = []

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)

        intersection = np.logical_and(pred_cls, label_cls).sum()
        union = np.logical_or(pred_cls, label_cls).sum()

        iou = np.nan if union == 0 else intersection / union
        ious.append(iou)

    ious = np.array(ious)
    mean_iou = np.nanmean(ious)
    metrics = {"mean_iou": mean_iou}
    for cls, iou in enumerate(ious):
        metrics[f"iou_class_{cls}"] = iou

    return metrics


def compute_mean_dice(pred, label, num_classes):
    """
    pred: (N, H, W) predicted class indices
    label: (N, H, W) ground-truth class indices
    num_classes: int
    """

    # Move to cpu + numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    dice_scores = []

    for c in range(num_classes):
        pred_c = (pred == c)
        label_c = (label == c)

        intersection = np.sum(pred_c & label_c)
        sum_pred = np.sum(pred_c)
        sum_label = np.sum(label_c)

        if sum_pred + sum_label == 0:
            # No pixels of this class at all → ignore this class
            continue

        dice = (2 * intersection) / (sum_pred + sum_label)
        dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    return float(np.mean(dice_scores))


def compute_pixel_accuracy(pred, label):
    """
    pred: (N, H, W) predicted class indices
    label: (N, H, W) ground-truth class indices
    """

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    correct = np.sum(pred == label)
    total = pred.size

    return float(correct / total)


def compute_metrics(p):
        """
        Compute per-class IoU and mean IoU for semantic segmentation predictions.

        p.predictions: (batch_size, num_classes, H_pred, W_pred)
        p.label_ids:   (batch_size, H_lbl, W_lbl)
        """
        if isinstance(p, dict):
            preds = p['predictions'].cpu()  # raw logits
            labels = p['label_ids'].cpu()   # ground-truth labels
        else:
            preds = p.predictions  # raw logits
            labels = p.label_ids   # ground-truth labels

        # Resize predictions to match label size
        logits = torch.nn.functional.interpolate(
            torch.tensor(preds),
            size=labels.shape[-2:],   # (H_lbl, W_lbl)
            mode="bilinear",
            align_corners=False
        )

        # Final predicted class mask
        pred = logits.argmax(dim=1).cpu().numpy()  # (batch_size, H_lbl, W_lbl)

        # compute ious
        metrics = compute_iou(pred, labels, num_classes=preds.shape[1])
        metrics['pa'] = compute_pixel_accuracy(pred, labels)
        metrics['mean_dice'] = compute_mean_dice(pred, labels, preds.shape[1])

        return metrics