import torch
from transformers import Trainer
from utils.metrics import compute_metrics


class TrainValMetricsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Buffers for storing batch results during an epoch
        self.training_metrics = []
        self.training_losses = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Standard HF forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Compute logits without affecting backward
        with torch.no_grad():
            logits = outputs.logits
            labels = inputs["labels"]

        # Save them for end-of-epoch metrics
        dict_for_metrics = {'predictions': logits.cpu(), "label_ids": labels.cpu()}
        metrics = compute_metrics(dict_for_metrics)
        self.training_metrics.append(metrics)
        self.training_losses.append(loss.cpu().detach().numpy())

        # Standard HF backward
        self.accelerator.backward(loss)
        return loss.detach()
    

# class TrainValMetricsTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.train_iou_list = []  # buffer to store batch IoU

#     def training_step(self, model, inputs, *args, **kwarg):
#         # Standard forward pass and loss computation
#         loss = super().training_step(model, inputs, *args, **kwarg)

#         # Compute batch metrics without affecting gradients
#         with torch.no_grad():
#             logits = model(**inputs).logits
#             labels = inputs["labels"]

#             logits_upsampled = torch.nn.functional.interpolate(
#                 logits,
#                 size=labels.shape[-2:],
#                 mode="bilinear",
#                 align_corners=False
#             )
#             preds = logits_upsampled.argmax(dim=1)
#             batch_iou = self.compute_mean_iou(preds, labels)

#             self.train_iou_list.append(batch_iou)

#         # Log averaged IoU every logging_steps
#         if (self.state.global_step + 1) % self.args.logging_steps == 0:
#             avg_iou = sum(self.train_iou_list) / len(self.train_iou_list)
#             self.log({"train_mean_iou": avg_iou})
#             self.train_iou_list.clear()  # reset buffer

#         return loss


#     def compute_mean_iou(self, preds, labels):
#         """Computes mean IoU for a batch (binary or multi-class)."""
#         num_classes = preds.max().item() + 1
#         ious = []

#         for cls in range(num_classes):
#             pred_mask = preds == cls
#             true_mask = labels == cls

#             intersection = (pred_mask & true_mask).sum().item()
#             union = (pred_mask | true_mask).sum().item()

#             if union == 0:
#                 continue

#             ious.append(intersection / union)

#         return sum(ious) / len(ious) if ious else 0.0
