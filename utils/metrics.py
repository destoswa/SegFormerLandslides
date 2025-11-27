import numpy as np
import torch


def compute_iou(preds, labels):

     assert len(preds) == len(labels)

     int = preds == labels
     
     return 1/((len(preds)+len(labels))/np.sum(int) -1)


def compute_mean_iou(preds, labels):
    num_classes = preds.shape[1]
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
    return mean_iou


def compute_metrics(p):
        """
        Compute per-class IoU and mean IoU for semantic segmentation predictions.

        p.predictions: (batch_size, num_classes, H_pred, W_pred)
        p.label_ids:   (batch_size, H_lbl, W_lbl)
        """
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

        num_classes = preds.shape[1]
        ious = []

        for cls in range(num_classes):
            pred_cls = (pred == cls)
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