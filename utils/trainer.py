import torch
import numpy as np
from transformers import Trainer
from utils.metrics import compute_metrics


class TrainValMetricsTrainer(Trainer):
    def __init__(self, confmat_dir, confmat_buffer_size=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Buffers for storing batch results during an epoch
        self.training_metrics = []
        self.training_losses = []
        self.confmat_dir = confmat_dir
        self.confmat = []
        self.confmat_idx = 0
        self.confmat_buffer_size = confmat_buffer_size

    @staticmethod
    def logits_to_preds(logits, height=512, width=512):
        # Resize predictions to match label size
        logits = torch.nn.functional.interpolate(
            torch.tensor(logits),
            size=(logits.shape[-2]*4, logits.shape[-1]*4),   # (H_lbl, W_lbl)
            mode="bilinear",
            align_corners=False
        )

        # Final predicted class mask
        return logits.argmax(dim=1).cpu().numpy()  # (batch_size, H_lbl, W_lbl)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Standard HF forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Compute logits without affecting backward
        with torch.no_grad():
            logits = outputs.logits
            labels = inputs["labels"]

        logits = logits.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = self.logits_to_preds(logits)
        # print(logits.shape)
        # print(labels.shape)
        # print(preds.shape)

        # Save them for end-of-epoch metrics
        dict_for_metrics = {'predictions': preds, "label_ids": labels}
        self.confmat.append(dict_for_metrics)
        self.confmat_idx += 1

        if len(self.confmat) * self.args.per_device_train_batch_size > self.confmat_buffer_size:
            #TODO save and clear
            pass


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
